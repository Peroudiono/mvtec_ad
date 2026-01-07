import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms

import streamlit as st

# =====================================================================================
# Paths & device
# =====================================================================================

PROJECT_ROOT = Path(r"C:\Users\othni\Projects\mvtec_ad")

DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
AD_MODELS_DIR = MODELS_DIR / "ad_resnet_mahalanobis"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Catégories MVTec (ordre alphabétique)
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

# Catégories où AD est très fort en mode SAFE (prec_safe >= 0.9 dans tes résultats)
STRONG_AD_CATEGORIES = ["bottle", "toothbrush", "metal_nut"]


# =====================================================================================
# Chargement modèles + seuils (cachés par Streamlit pour éviter de recharger à chaque run)
# =====================================================================================

@st.cache_resource
def load_models_and_thresholds():
    # ---------- Seuils globaux image-level ----------
    th_path = EXPERIMENTS_DIR / "image_level_thresholds.json"
    with open(th_path, "r") as f:
        th = json.load(f)

    tau_img_F1 = th["image_level_global"]["tau_F1"]
    tau_img_safe = th["image_level_global"]["tau_safe"]

    # ---------- Seuils AD par catégorie ----------
    ad_summary_path = EXPERIMENTS_DIR / "ad_resnet_mahalanobis_mvtec_all.json"
    with open(ad_summary_path, "r") as f:
        ad_data = json.load(f)

    # df_ad : index = ['auc_test','tau_F1','tau_F2','tau_safe',...]
    # colonnes = catégories
    df_ad = pd.DataFrame(ad_data)

    # Seuil AD "SAFE" (0 FN sur le test) par catégorie
    TAU_AD_SAFE = {
        cat: float(df_ad[cat]["tau_safe"])
        for cat in df_ad.columns
    }

    # Seuil AD "F1-optimal" par catégorie (utile pour PRECISION)
    TAU_AD_F1 = {
        cat: float(df_ad[cat]["tau_F1"])
        for cat in df_ad.columns
    }

    # ---------- Modèle de classification (ResNet18) ----------
    cls_model = models.resnet18(weights=None)
    in_features = cls_model.fc.in_features
    cls_model.fc = nn.Linear(in_features, 1)

    cls_ckpt = MODELS_DIR / "resnet18_image_level_best.pt"
    cls_model.load_state_dict(torch.load(cls_ckpt, map_location=device))
    cls_model = cls_model.to(device)
    cls_model.eval()

    # ---------- Extracteur de features pour AD (ResNet18 pré-entraîné ImageNet) ----------
    resnet_feat = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    modules = list(resnet_feat.children())[:-1]  # on enlève la FC
    resnet_feat = nn.Sequential(*modules).to(device)
    resnet_feat.eval()

    # ---------- Transform commun ----------
    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return (
        cls_model,
        resnet_feat,
        eval_transform,
        tau_img_F1,
        tau_img_safe,
        TAU_AD_SAFE,
        TAU_AD_F1,
    )


(
    cls_model,
    resnet_feat,
    eval_transform,
    tau_img_F1,
    tau_img_safe,
    TAU_AD_SAFE,
    TAU_AD_F1,
) = load_models_and_thresholds()


# =====================================================================================
# Chargement du CSV de référence MVTec (pour TP / FP / TN / FN)
# =====================================================================================

DF_SCORES_CSV = EXPERIMENTS_DIR / "df_test_all_with_scores.csv"
if DF_SCORES_CSV.exists():
    df_ref = pd.read_csv(DF_SCORES_CSV)
    # On ajoute une colonne "filename" = nom de fichier (000.png, etc.)
    df_ref["filename"] = df_ref["path"].apply(lambda p: Path(p).name)
else:
    df_ref = pd.DataFrame()  # vide → pas de type d'erreur possible


def get_error_type(category: str, filename: str, y_pred: int):
    """
    Retourne (y_true, err_type) avec err_type dans {TP, FP, TN, FN}
    en se basant sur df_test_all_with_scores.csv.

    Si l'image n'est pas trouvée (image perso, etc.) → (None, None).
    """
    if df_ref is None or df_ref.empty:
        return None, None

    # On matche sur (category, filename)
    row = df_ref[(df_ref["category"] == category) &
                 (df_ref["filename"] == filename)]

    if row.empty:
        return None, None

    y_true = int(row["label"].iloc[0])  # 0 = normal, 1 = anomalie

    if y_true == 1 and y_pred == 1:
        err_type = "TP"
    elif y_true == 0 and y_pred == 0:
        err_type = "TN"
    elif y_true == 1 and y_pred == 0:
        err_type = "FN"
    else:
        err_type = "FP"

    return y_true, err_type


# =====================================================================================
# Scoring d'une image : cls_prob + ad_score
# =====================================================================================

def compute_scores(img_pil: Image.Image, category: str):
    """
    Calcule :
      - cls_prob : sigmoïde du logit du classifieur ResNet
      - ad_score : distance de Mahalanobis aux normales de la catégorie
    """
    # 1) Passage par le transform
    x = eval_transform(img_pil).unsqueeze(0).to(device)  # (1,3,224,224)

    # 2) Score classification
    with torch.no_grad():
        logit = cls_model(x).squeeze(1)       # (1,)
        cls_prob = torch.sigmoid(logit).item()

    # 3) Score AD (Mahalanobis)
    npz_path = AD_MODELS_DIR / f"{category}_gaussian_stats.npz"
    data = np.load(npz_path)
    mu = data["mu"]
    precision = data["precision"]

    with torch.no_grad():
        feat = resnet_feat(x)                 # (1,512,1,1)
        feat = feat.view(1, -1).cpu().numpy() # (1,512)

    diff = feat - mu
    left = diff @ precision
    m = float(np.sqrt(np.sum(left * diff, axis=1))[0])

    return cls_prob, m


# =====================================================================================
# Policies de décision
# =====================================================================================

def apply_policy(category: str, cls_prob: float, ad_score: float, mode: str):
    """
    Retourne (y_pred, reason) avec :
      y_pred = 1 -> anomalie
      y_pred = 0 -> normal
    """

    tau_ad_safe = TAU_AD_SAFE[category]
    tau_ad_F1 = TAU_AD_F1[category]

    # -------- Mode 1 : AD_SAFE_ONLY (SAFE_0FN) ----------
    if mode == "SAFE_0FN":
        y = int(ad_score >= tau_ad_safe)
        reason = (
            f"Mode SAFE_0FN : y=1 (anomalie) si ad_score >= tau_ad_safe({tau_ad_safe:.3f})."
        )
        return y, reason

    # -------- Mode 2 : INDUSTRIAL_SAFE_0FN ----------
    if mode == "INDUSTRIAL_SAFE_0FN":
        if category in STRONG_AD_CATEGORIES:
            y = int(ad_score >= tau_ad_safe)
            reason = (
                "Mode INDUSTRIAL_SAFE_0FN : catégorie forte → "
                f"y=1 si ad_score >= tau_ad_safe({tau_ad_safe:.3f})."
            )
        else:
            # anomalie si AD_SAFE OU cls_prob suffisamment élevée (super safe)
            y = int((ad_score >= tau_ad_safe) or (cls_prob >= tau_img_safe))
            reason = (
                "Mode INDUSTRIAL_SAFE_0FN : catégorie non forte → "
                f"y=1 si ad_score >= tau_ad_safe({tau_ad_safe:.3f}) "
                f"OU cls_prob >= tau_img_safe({tau_img_safe:.3f})."
            )
        return y, reason

    # -------- Mode 3 : INDUSTRIAL_BALANCED ----------
    if mode == "INDUSTRIAL_BALANCED":
        if category in STRONG_AD_CATEGORIES:
            y = int(ad_score >= tau_ad_safe)
            reason = (
                "Mode INDUSTRIAL_BALANCED (catégorie forte) : "
                f"y=1 si ad_score >= tau_ad_safe({tau_ad_safe:.3f})."
            )
        else:
            y = int((ad_score >= tau_ad_safe) and (cls_prob >= 0.6))
            reason = (
                "Mode INDUSTRIAL_BALANCED (catégorie non forte) : "
                "y=1 si AD_SAFE (ad_score >= tau_ad_safe) "
                "ET cls_prob >= 0.60."
            )
        return y, reason

    # -------- Mode 4 : PRECISION ----------
    if mode == "PRECISION":
        y = int((ad_score >= tau_ad_F1) and (cls_prob >= tau_img_F1))
        reason = (
            "Mode PRECISION : y=1 si "
            f"ad_score >= tau_F1_cat({tau_ad_F1:.3f}) "
            f"ET cls_prob >= tau_img_F1({tau_img_F1:.3f})."
        )
        return y, reason

    # Fallback
    y = int(ad_score >= tau_ad_safe)
    reason = (
        f"Mode inconnu '{mode}', fallback AD_SAFE_ONLY : "
        f"y=1 si ad_score >= tau_ad_safe({tau_ad_safe:.3f})."
    )
    return y, reason


# =====================================================================================
# Interface Streamlit
# =====================================================================================

st.set_page_config(page_title="MVTec AD – Image-level", layout="wide")

st.title("🔍 Détection d’anomalies sur MVTec AD – Image level (ResNet + Mahalanobis)")

st.markdown(
    """
Ce mini outil Streamlit utilise ton **modèle ResNet image-level** et le score
**AD Mahalanobis** par catégorie pour décider si une image est *normale* ou *anormale*.

Tu peux :

1. Choisir un **mode de décision** (policy),
2. Choisir la **catégorie MVTec**,
3. Uploader une image (MVTec ou autre),
4. Voir la décision + les scores bruts (cls_prob, ad_score) et, si possible, le type
   de prédiction (TP / FP / TN / FN) par rapport au dataset MVTec.
"""
)

# --- Sidebar : choix du mode et de la catégorie ---
st.sidebar.header("⚙️ Paramètres")

mode = st.sidebar.selectbox(
    "Mode de décision",
    [
        "SAFE_0FN",
        "INDUSTRIAL_SAFE_0FN",
        "INDUSTRIAL_BALANCED",
        "PRECISION",
    ],
    index=1,  # par défaut INDUSTRIAL_SAFE_0FN
)

category = st.sidebar.selectbox("Catégorie MVTec", CATEGORIES, index=0)

show_details = st.sidebar.checkbox("Afficher les détails des seuils", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Rappel rapide :**\n\n"
    "- `SAFE_0FN` : priorité au rappel (0 faux négatifs sur MVTec test).\n"
    "- `INDUSTRIAL_SAFE_0FN` : proche de `SAFE_0FN`, mais utilise aussi le classifieur sur certaines catégories.\n"
    "- `INDUSTRIAL_BALANCED` : compromis précision / rappel.\n"
    "- `PRECISION` : policy la plus stricte (précision maximale)."
)

# --- Upload d'image ---
uploaded_file = st.file_uploader(
    "📁 Choisis une image (PNG/JPG/JPEG) :", type=["png", "jpg", "jpeg"]
)

col_img, col_res = st.columns([1, 1.2])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    with col_img:
        st.image(img, caption="Image d'entrée", use_column_width=True)

    # --- Scoring ---
    cls_prob, ad_score = compute_scores(img, category)
    y_pred, reason = apply_policy(category, cls_prob, ad_score, mode)

    label_text = "🚨 Anomalie" if y_pred == 1 else "✅ Normal"

    # On essaie de retrouver le type de prédiction (TP/FP/TN/FN) si l'image vient de MVTec
    filename = Path(uploaded_file.name).name
    y_true, err_type = get_error_type(category, filename, y_pred)

    with col_res:
        st.subheader("Résultat du modèle")

        st.markdown(f"**Décision : {label_text} (y = {y_pred})**")
        st.markdown(f"**Policy utilisée :** `{mode}`")
        st.markdown(f"**Raison :** {reason}")

        # Info sur le type de prédiction si on connaît la vérité terrain
        if y_true is not None:
            txt_true = "Anomalie (1)" if y_true == 1 else "Normal (0)"
            st.markdown(f"**Étiquette vraie (MVTec) :** {txt_true}")
            st.markdown(f"**Type de prédiction :** `{err_type}`")  # TP / FP / TN / FN
        else:
            st.markdown(
                "*Type de prédiction (TP/FP/TN/FN) indisponible : "
                "image hors dataset MVTec ou non trouvée dans le CSV.*"
            )

        st.markdown("---")
        st.markdown("### Scores bruts")

        st.write(f"- Probabilité classifieur (ResNet)  : `{cls_prob:.4f}`")
        st.write(f"- Score AD (Mahalanobis)           : `{ad_score:.4f}`")

        if show_details:
            tau_safe_cat = TAU_AD_SAFE[category]
            tau_f1_cat = TAU_AD_F1[category]

            st.markdown("### Seuils utilisés")
            st.write(f"- `tau_img_F1` (global)           : `{tau_img_F1:.4f}`")
            st.write(f"- `tau_img_safe` (global)         : `{tau_img_safe:.4f}`")
            st.write(f"- `tau_ad_safe[{category}]`       : `{tau_safe_cat:.4f}`")
            st.write(f"- `tau_ad_F1[{category}]`         : `{tau_f1_cat:.4f}`")
            st.write(f"- Catégorie forte AD ?            : `{category in STRONG_AD_CATEGORIES}`")

else:
    st.info("Upload une image pour lancer une prédiction.")

# (.venv) PS C:\Users\othni\Projects\mvtec_ad> python -m streamlit run app_image_level.py