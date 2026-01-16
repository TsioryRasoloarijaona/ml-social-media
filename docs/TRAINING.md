# Entraînement du modèle (train_v2.ipynb)

Ce document décrit brièvement la théorie derrière XGBoost, explique les hyperparamètres utilisés, décrit la procédure d'entraînement employée dans `rebuild/scripts/train_v2.ipynb` et indique où trouver les résultats (métriques, graphiques et modèle sauvegardé).

---

## 1. Récapitulatif rapide

- Jeu de données utilisé : `final_v2.csv` (prétraité depuis `data/final_v2.csv`).
- Target : `ER` (taux d'engagement) transformée avec `log1p` (y = log1p(ER)).
- Split : `train_test_split(test_size=0.2, random_state=42)`.
- Modèle : `xgboost.XGBRegressor` avec les paramètres appliqués dans le notebook (voir section 4).
- Artéfacts générés : graphiques (`scatter_plot.png`, `params.png`) dans `/graphs` et modèle(s) dans `rebuild/model/`.

---

## 2. Formules théoriques essentielles de XGBoost

XGBoost est un algorithme de boosting de gradient basé sur des arbres. Les points clefs (formules simplifiées) :

- Fonction objectif régulière (sur T arbres) :

  L(\theta) = \sum_{i} l(y_i, \hat y_i) + \sum_{k=1}^{T} \Omega(f_k)

  où l() est la loss (ex. squared error) et

  \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2

  (T = nombre de feuilles, w_j = poids d'une feuille).

- Approximation au second ordre : on utilise la dérivée première (gradient) g_i et deuxième dérivée (hessienne) h_i de la loss par rapport à la prédiction :

  g_i = \partial_{\hat y_i} l(y_i, \hat y_i), \quad h_i = \partial^2_{\hat y_i} l(y_i, \hat y_i)

- Poids optimal d'une feuille (après avoir regroupé les gradients/hessians des exemples de la feuille) :

  w^* = - \dfrac{G}{H + \lambda}

  où G = \sum_{i\in feuille} g_i, H = \sum_{i\in feuille} h_i.

- Gain (amélioration) d'une séparation (split) entre gauche (L) et droite (R) :

  Gain = \tfrac{1}{2} \left(\dfrac{G_L^2}{H_L + \lambda} + \dfrac{G_R^2}{H_R + \lambda} - \dfrac{G_{L+R}^2}{H_{L+R} + \lambda}\right) - \gamma

  Un split est effectué seulement si le Gain est positif et dépasse le seuil `gamma`.

- Shrinkage (learning rate) : après calcul des poids optimaux, on applique un facteur d'atténuation `eta` (learning_rate) :

  w_{new} = eta * w^*

  Ceci réduit la contribution de chaque itération pour améliorer la généralisation.

---

## 3. Hyperparamètres — sens et recommandations

Ci-dessous les hyperparamètres fréquemment utilisés et ceux présents dans `train_v2.ipynb`.

- `n_estimators` : nombre d'arbres (boosting rounds). Plus grand -> plus de capacité. Recommandation : débuter 100–1000 et utiliser `early_stopping` si possible.

- `learning_rate` (eta) : taux d'apprentissage (shrinkage). Valeurs communes : 0.01–0.3. Plus petit -> nécessite plus d'arbres mais réduit l'overfitting.

- `max_depth` : profondeur maximale des arbres. Contrôle la complexité. Valeurs typiques : 3–10.

- `subsample` : fraction d'échantillons (rows) utilisés pour construire chaque arbre. < 1.0 injecte du bruit pour réduire l'overfitting (ex. 0.5–1.0).

- `colsample_bytree` : fraction de features échantillonnées pour chaque arbre. Aide à réduire la corrélation entre arbres (0.3–1.0).

- `gamma` : réduction minimale de la loss pour qu'un split soit considéré. Plus grand -> modèle plus conservateur.

- `min_child_weight` : poids (somme des hessians) minimal requis dans une feuille pour effectuer un split. Empêche les feuilles avec trop peu d'exemples.

- `reg_lambda` (λ) : régularisation L2 sur les poids des feuilles. Stabilise les poids.

- `reg_alpha` : régularisation L1 (pousse vers la parcimonie des poids).

- `objective` : fonction de perte (ex. `reg:squarederror` pour régression).

- `random_state` / `seed` : graine pour reproductibilité.

Conseil de tuning (ordre pratique) :
1) Fixer un `learning_rate` faible (ex. 0.05) et augmenter `n_estimators` ; utiliser early stopping.  
2) Ajuster `max_depth` et `min_child_weight`.  
3) Ajuster `subsample` et `colsample_bytree`.  
4) Régularisations `reg_alpha`/`reg_lambda` si overfitting persiste.

---

## 4. Procédure d'entraînement observée dans `train_v2.ipynb`

Résumé des étapes et des paramètres exacts utilisés dans le notebook fourni :

1. Lecture des données :
   - `df = pd.read_csv('social_media_cleaned.csv')`

2. Préparation :
   - Séparation X / y : `X = df.drop(columns=['ER'])`, `y = df['ER'].apply(np.log1p)` (transformation log1p sur la target).
   - Split : `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`.

3. Instanciation du modèle :

```python
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

4. Entraînement : `model.fit(X_train, y_train)`.

5. Évaluation :
   - Prédictions sur `X_test` : `y_pred = model.predict(X_test)`.
   - Calculs affichés dans le notebook :
     - R2 : `r2_score(y_test, y_pred)`
     - RMSE : `np.sqrt(mean_squared_error(y_test, y_pred))`

6. Visualisations sauvegardées (dans `/graphs`) :
   - `scatter_plot.png` — Scatter plot Réel vs Prédit (y en log1p scale dans le notebook).
   - `params.png` — Importance des features (top features).

7. Sauvegarde du modèle : `joblib.dump(model, 'model.pkl')` (des modèles peuvent aussi être présents sous `rebuild/model/` comme `model_v2.pkl`, etc.).

---

## 5. Résultats — où les trouver et comment les reproduire

Les artefacts générés par l'entraînement sont déjà présents dans le dépôt :

- Graphiques (dossier `/graphs`) :
  - `/graphs/scatter_plot.png`  — Réel vs Prédit (visualisation de la qualité des estimations).
  - `/graphs/params.png`        — Importance des paramètres (feature importance).
  - `/graphs/boxplot_ER_reparition.png` — utilisé durant la transformation des données.

- Modèles (dossier `rebuild/model/`) :
  - `rebuild/model/model_v2.pkl`, `model_v3.pkl`, `model_v4.pkl`, `model.pkl` (vérifier lequel correspond à l'entraînement courant).

- Notebook d'entraînement : `rebuild/scripts/train_v2.ipynb`.

Si vous souhaitez ré-exécuter l'évaluation et afficher les métriques (R2, RMSE) localement, exécutez le script suivant dans l'environnement du projet (assurez-vous d'avoir installé les dépendances listées dans `requirements.txt`) :

```python
# Évaluer un modèle sauvegardé et afficher R2 / RMSE
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error

# Chemins possibles du modèle — choisissez le bon si nécessaire
model_paths = [
    'rebuild/model/model_v2.pkl',
    'rebuild/model/model.pkl'
]

# Charger les données pré-traitées (même fichier que celui utilisé pour l'entraînement)
df = pd.read_csv('social_media_cleaned.csv')
X = df.drop(columns=['ER'])
y = df['ER'].apply(np.log1p)

# Split identique à l'entraînement
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Charger le premier modèle existant
for p in model_paths:
    try:
        model = joblib.load(p)
        print(f"Loaded model from {p}")
        break
    except Exception:
        model = None

if model is None:
    raise FileNotFoundError('Aucun modèle trouvé aux emplacements listés. Vérifiez les chemins.')

# Prédictions et métriques
y_pred = model.predict(X_test)
print(f"Score R2 Final : {r2_score(y_test, y_pred):.4f}")
print(f"RMSE Final : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Option: afficher les graphiques enregistrés
print('\nGraphiques disponibles dans /graphs :')
import os

for f in os.listdir('../graphs'):
    if f.endswith('.png'):
        print(' -', os.path.join('../graphs', f))
```

Notes :
- Le notebook original effectue la transformation `log1p` sur la target avant l'entraînement. Si vous souhaitez interpréter les prédictions en unité originale, appliquez `np.expm1` sur `y_pred` et `y_test`.
- Assurez-vous que `social_media_cleaned.csv` correspond bien à celle utilisée pour l'entraînement (mêmes colonnes / encodages).

---



---

## 7. Emplacements utiles dans le dépôt

- Notebook d'entraînement : `rebuild/scripts/train_v2.ipynb`
- Graphiques : `graphs/` (ex. `graphs/scatter_plot.png`, `graphs/params.png`)
- Modèles sauvegardés : `rebuild/model/`
- Données pré-traitées : `social_media_cleaned.csv` (à la racine du repo selon le notebook)

