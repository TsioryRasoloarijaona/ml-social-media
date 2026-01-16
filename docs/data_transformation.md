# Transformation pipeline — `rebuild/scripts/trasformation_v2.ipynb`

## Objectif
Transformer les données brutes de `../data/social_media.csv` pour produire un jeu de données nettoyé et feature-engineered enregistré dans `../data/final.csv`.

## Pré-requis
- Python avec les bibliothèques: `pandas`, `matplotlib`, `seaborn`, `textblob`, `scikit-learn`, `joblib`, `numpy`.
- Kernel du notebook lancé (les imports sont dans la première cellule).

## Entrées / Sorties
- Entrée: `../data/social_media.csv`
- Sortie: `../data/final.csv`
- Graphique sauvegardé: `boxplot_ER_reparition.png`

## Étapes et changements (bref)

1. Import des bibliothèques  
   - `pandas` (`pd`), `matplotlib.pyplot` (`plt`), `seaborn` (`sns`), `TextBlob`, `LabelEncoder`, `joblib` (`jb`), `numpy` (`np`).

2. Lecture du fichier  
   - Chargement CSV: `df = pd.read_csv('../data/social_media.csv')`.

3. Filtrage de plateforme  
   - Garder uniquement les lignes où `platform == 'Instagram'`.  
   - Supprimer la colonne `platform`.

4. Calcul du taux d'engagement (ER)  
   - ER = ((`comments_count` + `likes` + `shares` + `views`) / `follower_count`) * 100.  
   - Suppression des lignes où `ER > 100` (valeurs manifestement erronées).

5. Détection et suppression d'outliers sur ER  
   - Calcul de `mean` et `std` de `ER`.  
   - Limite haute = `mean + 3*std`. Suppression des lignes au-delà de cette limite.

6. Visualisation  
   - Boxplot de la distribution de `ER` (normalisé par 100 dans le tracé).  
   - Sauvegarde de l'image `boxplot_ER_reparition.png`.

7. Extraction temporelle  
   - Création de `day` (nom du jour) et `hour` à partir de `post_date` via `pd.to_datetime`.

8. Score de sentiment et compte de mots / hashtags  
   - `description_score` = polarité TextBlob appliquée à `content_description` (0 si NA).  
   - `hashtag_count` = nombre d'items séparés par des virgules dans `hashtags` (0 si NA ou chaîne vide).  
   - `desciption_word_count` = nombre de mots dans `content_description` (0 si NA).  
  
9. Encodage cyclique du jour  
   - Mappe les jours en nombres 0..6 (`day_num`).  
   - Ajoute `day_sin` et `day_cos` via `np.sin` et `np.cos` pour l'encodage cyclique hebdomadaire.

10. Encodage catégoriel et booléens  
    - `pd.get_dummies` sur `content_type`, `drop_first=True`.  
    - Conversion des colonnes booléennes en `int`.

11. Nettoyage final des colonnes inutiles  
    - Suppression: `post_date`, `content_description`, `likes`, `views`, `shares`, `hashtags`, `day_num`, `day`, `comments_count`.

12. Export  
    - Sauvegarde CSV: `df.to_csv('../data/final.csv', index=False)`.


