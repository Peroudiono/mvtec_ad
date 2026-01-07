# MVTec AD — Industrial Anomaly Detection Pipeline (ResNet + Mahalanobis + YOLO + Policies)

Un pipeline **end-to-end** de détection d’anomalies industrielles sur **MVTec AD** avec une contrainte “métier” forte : **minimiser au maximum les faux négatifs** (ne pas laisser passer un défaut), tout en gardant des **politiques de décision** capables de réduire les faux positifs selon le contexte.

Le projet combine :
- **Classification image-level** (ResNet-18) : *good* vs *defect*
- **Anomaly Detection** (features ResNet + **distance de Mahalanobis**) : score d’anomalie *par catégorie*
- **Localisation** des défauts (**YOLOv8n fine-tuné**) : bounding boxes + labels
- **Policies** (règles + seuils) : modes *SAFE*, *BALANCED*, *PRECISION*, etc.
- **App Streamlit** : démo et inspection visuelle

---

## Objectif principal

En industrie, **un faux négatif** peut coûter très cher (sécurité, qualité, retours, réputation).  
Ce dépôt met donc l’accent sur :
- **calibration** des seuils (F1 / F2 / SAFE),
- **cascades** (AD puis classifieur ou l’inverse),
- **politiques décisionnelles** paramétrables via JSON (sans ré-entraîner).

---

## Idée clé (en 1 phrase)

> On apprend à reconnaître “le comportement normal” (features des *good*), puis on mesure à quel point une nouvelle image s’en éloigne (Mahalanobis), et on combine ça avec un classifieur + un détecteur YOLO selon une politique.

---

## Architecture (vue simple)

```
Image
  │
  ├── ResNet-18 (classification) ──► prob defect p(x) ──► seuillage (τ_img)
  │
  ├── ResNet-18 (feature extraction) ─► φ(x) ∈ R^512 ─► Mahalanobis s(x) ─► τ_ad(cat)
  │
  └── YOLOv8n (detection) ─────────► bboxes + labels (localisation)
                         │
                         ▼
                 POLICIES (SAFE / BALANCED / PRECISION)
                         │
                         ▼
                 Décision finale + explication
```

---

## Structure du dépôt

```
mvtec_ad/
├─ app.py                       # démo (Streamlit) / pipeline bottle (selon ta version)
├─ app_image_level.py           # app / API image-level multi-catégories (selon ta version)
├─ prepare_mvtec_yolo_all.py    # préparation des données YOLO multi-catégories
├─ mvtec_yolo_all.yaml          # YAML Ultralytics (classes, chemins, splits)
├─ experiments/                 # CSV, JSON, seuils, résultats, caches
├─ models/                      # checkpoints ResNet, stats gaussiennes AD, etc.
├─ notebooks/                   # notebooks du pipeline (00 → 08)
├─ runs/                        # sorties Ultralytics / entraînements YOLO
└─ data/                        # dataset MVTec (non inclus)
```

> **Le dataset n’est pas inclus**. MVTec AD a sa propre licence. Télécharge-le séparément et place-le dans `data/`.

---

## Prérequis

- Python 3.9+ (3.10/3.11 OK)
- GPU optionnel (CPU possible mais plus lent)
- Bibliothèques principales :
  - `torch`, `torchvision`
  - `numpy`, `pandas`, `scikit-learn`
  - `matplotlib`, `opencv-python`, `Pillow`
  - `ultralytics` (YOLO)
  - `streamlit` (app)

### Créer un environnement virtuel (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### Installer les dépendances

Si tu as un `requirements.txt` :
```bash
pip install -r requirements.txt
```

Sinon, version rapide :
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib opencv-python pillow ultralytics streamlit
```

> Astuce : générer ton `requirements.txt` :
```bash
pip freeze > requirements.txt
```

---

## Dataset : MVTec AD

- `train/good/` : normales (apprentissage “normalité”)
- `test/` : good + défauts (types variés)
- `ground_truth/` : masques binaires (défauts uniquement)

Tu peux garder la structure originale MVTec et pointer le chemin racine dans tes notebooks/scripts.

---

## Notebooks (pipeline recommandé)

Les notebooks sont organisés de façon progressive (les noms peuvent varier selon ta version, mais la logique reste la même) :

1. **00_check_mvtec_structure.ipynb**  
   Vérifie la structure du dataset, les catégories, les masques.

2. **01_build_image_level_dataframe.ipynb**  
   Construit l’index global `image_level_df.csv` (paths, category, split, label, defect_type).

3. **02_train_resnet18_image_level.ipynb**  
   Entraîne ResNet-18 binaire multi-catégories (*good* vs *defect*).  
   Sauvegarde un meilleur modèle (selon AUC val).

4. **03_threshold_sweep.ipynb**  
   Calibre les seuils `τ` : `τ_F1`, `τ_F2`, `τ_SAFE` (mode “0 FN sur val si possible”).

5. **04_mahalanobis_bottle.ipynb**  
   AD sur *bottle* : features ResNet + gaussienne (μ, Σ) + Mahalanobis.

6. **05_mahalanobis_all_categories.ipynb**  
   AD sur les **15 catégories** : stats par catégorie + seuils (F1/F2/SAFE) exportés en JSON.

7. **06_cascade_eval.ipynb**  
   Compare des policies (AD seul, OR, AND, mix par catégories “fortes”, etc.).

8. **07_inference_api.ipynb**  
   Construit une API d’inférence unifiée (score + décision + raison).

9. **08_error_analysis.ipynb**  
   Analyse fine des FP/FN et policies guidées par erreurs (par catégories, puis par couples (cat, defect_type)).

---

## 🔍 Anomaly Detection (Mahalanobis) — ce que ça fait

Pour une catégorie donnée :
1. On extrait des features `φ(x) ∈ R^512` pour les images **good** du train.
2. On estime la moyenne `μ` et la covariance `Σ` du “nuage normal”.
3. Pour une image test, on calcule un score :
   \[
   s(x) = \sqrt{(\phi(x)-\mu)^T \Sigma^{-1} (\phi(x)-\mu)}
   \]
4. Plus `s(x)` est grand, plus l’image est **suspecte**.
5. On choisit un seuil `τ_ad(cat)` via F1/F2/SAFE.

---

## Policies (décision finale)

Le dépôt utilise des **policies** (règles) qui combinent :
- `p_cls(x)` (probabilité defect du classifieur)
- `s_ad(x)` (score Mahalanobis par catégorie)
- (optionnel) YOLO (localisation)

Exemples typiques :
- **SAFE_0FN** : priorité absolue au rappel (zéro FN si possible), souvent AD_SAFE
- **PRECISION** : peu de FP, plus strict (AND + seuils plus hauts)
- **INDUSTRIAL_BALANCED** : mix intelligent (catégories “fortes” en AD, cascade ailleurs)
- **INDUSTRIAL_SAFE_0FN** : version “zéro FN” plus précise que AD seul via exceptions

Les seuils et règles sont généralement centralisés dans des fichiers JSON du type :
- `experiments/image_level_thresholds.json`
- `experiments/ad_resnet_mahalanobis_mvtec_all.json`
- `experiments/image_level_policies_config.json`

---

## YOLO (localisation des défauts)

Le pipeline YOLO sert à :
- **localiser** où est le défaut (bbox),
- ajouter une couche “explicable” en inspection.

### Préparation des données YOLO multi-catégories
Le script `prepare_mvtec_yolo_all.py` convertit (si besoin) masques → bboxes et prépare l’arborescence YOLO.

### Entraîner YOLOv8 (Ultralytics)
Exemple générique :
```bash
yolo detect train data=mvtec_yolo_all.yaml model=yolov8n.pt imgsz=256 epochs=50
```

Le meilleur modèle est généralement dans :
- `runs/detect/train*/weights/best.pt`

---

## Lancer l’application Streamlit

Selon ton usage (app bottle / app image-level) :

```bash
streamlit run app.py
```

ou

```bash
streamlit run app_image_level.py
```

---

## Résultats (interprétation)

Les métriques importantes :
- **Recall** (rappel) : “combien de défauts je capture”
- **Precision** : “quand je dis défaut, est-ce vrai”
- **F1 / F2** : compromis (F2 favorise plus le rappel)
- **AUROC** : séparabilité des scores (AD ou classifieur)

Le projet met volontairement en avant l’idée :
> **On choisit un seuil selon le risque métier**, pas “au hasard”.

---

## Roadmap (idées d’amélioration)

- Pixel-level AD : PatchCore / PaDiM / DRAEM (localisation fine)
- Calibration probabiliste : temperature scaling / isotonic
- Monitoring production : drift, seuils adaptatifs
- Auto-selection de policies selon coût FP/FN
- Meilleure stratification multi-catégories (équilibre par catégorie)

---

## Notes importantes

- MVTec AD n’est pas redistribué ici : tu dois le télécharger séparément.
- Les performances peuvent varier selon :
  - taille d’image (224/256),
  - augmentation,
  - CPU vs GPU,
  - seuils choisis.

---

## Auteur / Contact

- GitHub : `Matheux14`
- Projet : pipeline détection d’anomalies industrielles (MVTec AD)

Si tu veux, je peux aussi te générer :
- un **diagramme SVG/PNG** propre pour le README,
- un **requirements.txt** “propre”,
- une section **Reproductibilité** (commands exactes des notebooks).
