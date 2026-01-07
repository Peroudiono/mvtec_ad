import os
import random

import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO


# ------------------ Chargement du modèle (caché) ------------------ #
@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def run_inference(model: YOLO, image, conf: float, iou: float):
    """
    Lance l'inférence YOLO sur une image PIL et renvoie :
      - l'image annotée (numpy RGB)
      - un DataFrame avec les comptes par type de défaut
    """
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    res = results[0]

    # Image annotée (YOLO renvoie BGR -> on repasse en RGB)
    plotted = res.plot()[:, :, ::-1]

    # Récupération des classes détectées
    if res.boxes is None or res.boxes.cls is None:
        return plotted, pd.DataFrame(columns=["defect_type", "count"])

    class_ids = res.boxes.cls.cpu().numpy().astype(int)
    names = model.names

    labels = [names[int(cid)] for cid in class_ids]
    counts = (
        pd.Series(labels)
        .value_counts()
        .rename_axis("defect_type")
        .reset_index(name="count")
    )

    return plotted, counts


# ------------------------------ App ------------------------------ #
def main():
    st.set_page_config(
        page_title="MVTec YOLO – Visualisation des défauts",
        layout="wide"
    )

    st.title("🛠️ MVTec AD – Visualisation des types de défauts (YOLOv8)")
    st.markdown(
        "Cette interface te permet de **charger ton modèle YOLO** "
        "et de **visualiser les défauts détectés** sur les images."
    )

    # --------- Sidebar : paramètres --------- #
    with st.sidebar:
        st.header("⚙️ Paramètres")

        weights_path = st.text_input(
            "Chemin vers les poids YOLO (.pt)",
            value="runs/detect/train/weights/best.pt",  # à adapter si besoin
        )

        conf = st.slider("Seuil de confiance", 0.0, 1.0, 0.5, 0.01)
        iou = st.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.7, 0.05)

        st.markdown("---")
        st.markdown("**Option dataset MVTec**")

        dataset_root = st.text_input(
            "Dossier d'images (par ex. images de validation)",
            value="yolo_mvtec_all/images/val",  # à adapter si besoin
        )
        use_sample = st.checkbox(
            "Prendre une image aléatoire de ce dossier",
            value=False
        )

    # --------- Vérification des poids --------- #
    if not os.path.exists(weights_path):
        st.error(f"Fichier de poids introuvable : `{weights_path}`")
        st.stop()

    model = load_model(weights_path)

    # --------- Chargement de l'image --------- #
    img = None
    img_source = ""

    if use_sample:
        if os.path.isdir(dataset_root):
            all_images = [
                os.path.join(dataset_root, f)
                for f in os.listdir(dataset_root)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
            ]
            if not all_images:
                st.warning("Aucune image trouvée dans ce dossier.")
            else:
                img_path = random.choice(all_images)
                img = Image.open(img_path).convert("RGB")
                img_source = f"Image aléatoire : `{os.path.relpath(img_path, dataset_root)}`"
        else:
            st.warning("Le dossier spécifié pour le dataset n'existe pas.")

    else:
        uploaded = st.file_uploader(
            "📤 Upload une image MVTec (ou assimilée)",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        )
        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            img_source = f"Image uploadée : `{uploaded.name}`"

    if img is None:
        st.info("Charge une image (ou coche l’option d’échantillon aléatoire) pour commencer.")
        st.stop()

    st.markdown(f"**Source :** {img_source}")

    # --------- Inférence --------- #
    with st.spinner("Inférence YOLO en cours..."):
        plotted, counts = run_inference(model, img, conf=conf, iou=iou)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Image d’origine")
        st.image(img, use_column_width=True)

    with col2:
        st.subheader("📌 Détections YOLO (types de défauts)")
        st.image(plotted, use_column_width=True)

    # --------- Stats sur les défauts --------- #
    st.markdown("---")
    st.subheader("📊 Répartition des types de défauts détectés")

    if counts.empty:
        st.info("Aucun défaut détecté sur cette image (selon le seuil de confiance choisi).")
    else:
        # Filtre par type de défaut
        selected = st.multiselect(
            "Filtrer par type de défaut",
            options=counts["defect_type"].tolist(),
            default=counts["defect_type"].tolist(),
        )
        filtered = counts[counts["defect_type"].isin(selected)]

        st.caption("Table des défauts détectés :")
        st.dataframe(filtered, use_container_width=True)

        st.caption("Histogramme des défauts :")
        st.bar_chart(filtered.set_index("defect_type"))

    st.markdown("---")
    st.markdown("💡 Pour lancer l’application :")
    st.code("streamlit run app.py", language="bash")


if __name__ == "__main__":
    main()

# PS C:\Users\othni\Projects\mvtec_ad> & C:/Users/othni/Projects/mvtec_ad/.venv/Scripts/Activate.ps1
# (.venv) PS C:\Users\othni\Projects\mvtec_ad> python -m streamlit run app.py