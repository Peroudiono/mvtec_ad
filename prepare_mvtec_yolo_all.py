import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label
import yaml  # pip install pyyaml si besoin

# ------------------------------------------------------------------
# Chemins de base
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
YOLO_ROOT = PROJECT_ROOT / "yolo_mvtec_all"   # nouveau dossier YOLO

CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

random.seed(42)


def ensure_dirs():
    for split in ["train", "val"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def mask_to_bboxes(mask_path: Path, img_size):
    """
    Transforme un mask binaire MVTec en une liste de bounding boxes YOLO
    (x_center, y_center, w, h) normalisés.
    On sépare les composantes connexes (plusieurs défauts possibles).
    """
    w, h = img_size

    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask) > 0

    if not mask_np.any():
        return []

    labeled, num = cc_label(mask_np)
    bboxes = []

    for k in range(1, num + 1):
        ys, xs = np.where(labeled == k)
        if xs.size == 0:
            continue

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        bbox_w = x_max - x_min + 1
        bbox_h = y_max - y_min + 1
        x_c = x_min + bbox_w / 2.0
        y_c = y_min + bbox_h / 2.0

        # normalisation YOLO
        bboxes.append(
            (
                x_c / w,
                y_c / h,
                bbox_w / w,
                bbox_h / h,
            )
        )

    return bboxes


def main():
    ensure_dirs()

    class2id = {}  # "bottle_broken_large" -> 0, etc.

    for cat in CATEGORIES:
        print(f"\n=== Catégorie {cat} ===")
        cat_dir = DATA_ROOT / cat
        train_good_dir = cat_dir / "train" / "good"
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        # -------------------------------
        # 1) Images "good"
        #    - train/good -> split train
        #    - test/good  -> split val
        # -------------------------------
        for img_path in sorted(train_good_dir.glob("*.png")):
            dest_img = YOLO_ROOT / "images" / "train" / f"{cat}_good_{img_path.name}"
            dest_lbl = YOLO_ROOT / "labels" / "train" / (dest_img.stem + ".txt")

            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("")  # pas d'objets (image normale)

        good_test_dir = test_dir / "good"
        for img_path in sorted(good_test_dir.glob("*.png")):
            dest_img = YOLO_ROOT / "images" / "val" / f"{cat}_good_{img_path.name}"
            dest_lbl = YOLO_ROOT / "labels" / "val" / (dest_img.stem + ".txt")

            shutil.copy2(img_path, dest_img)
            dest_lbl.write_text("")

        # -------------------------------
        # 2) Images défectueuses
        #    test/<defect_type>/xxx.png
        #    ground_truth/<defect_type>/xxx_mask.png
        # -------------------------------
        for defect_dir in sorted(test_dir.iterdir()):
            defect_type = defect_dir.name
            if defect_type == "good" or not defect_dir.is_dir():
                continue

            class_name = f"{cat}_{defect_type}"
            if class_name not in class2id:
                class2id[class_name] = len(class2id)
            cid = class2id[class_name]

            imgs = sorted(defect_dir.glob("*.png"))
            if not imgs:
                continue

            random.shuffle(imgs)
            n = len(imgs)
            n_train = max(1, int(0.8 * n))  # 80% train, 20% val

            for i, img_path in enumerate(imgs):
                split = "train" if i < n_train else "val"

                img = Image.open(img_path)
                w, h = img.size

                mask_name = img_path.stem + "_mask" + img_path.suffix
                mask_path = gt_dir / defect_type / mask_name
                if not mask_path.exists():
                    print(f"  [WARN] mask manquant pour {img_path}")
                    continue

                bboxes = mask_to_bboxes(mask_path, (w, h))
                if not bboxes:
                    print(f"  [WARN] pas de pixels positifs dans {mask_path}")
                    continue

                dest_img = (
                    YOLO_ROOT
                    / "images"
                    / split
                    / f"{cat}_{defect_type}_{img_path.name}"
                )
                dest_lbl = (
                    YOLO_ROOT
                    / "labels"
                    / split
                    / (dest_img.stem + ".txt")
                )

                shutil.copy2(img_path, dest_img)

                lines = []
                for (xc, yc, bw, bh) in bboxes:
                    lines.append(
                        f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                    )
                dest_lbl.write_text("\n".join(lines))

        print(f"  -> terminé pour {cat}")

    # ------------------------------------------------------------------
    # 3) Écriture du YAML YOLO
    # ------------------------------------------------------------------
    names = {idx: name for name, idx in class2id.items()}
    yaml_dict = {
        "path": str(YOLO_ROOT),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }

    yaml_path = PROJECT_ROOT / "mvtec_yolo_all.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=True, allow_unicode=True)

    print("\n=== Résumé ===")
    print(f"Nb classes YOLO : {len(class2id)}")
    print("Classes (id -> name) :")
    for k in sorted(names.keys()):
        print(f"  {k:2d}: {names[k]}")
    print(f"\nDataset YOLO créé dans : {YOLO_ROOT}")
    print(f"Config YOLO écrite dans : {yaml_path}")


if __name__ == "__main__":
    main()
