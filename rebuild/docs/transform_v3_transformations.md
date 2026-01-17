# Transformations détaillées — `transform_v3.ipynb`

Ce document décrit, étape par étape, les transformations appliquées dans le notebook `rebuild/scripts/transform_v3.ipynb`. Pour chaque bloc : but, entrées, sorties, logique (formules, fonctions), et points d'attention.

---

## 0. Contexte
- Fichier source : `rebuild/scripts/transform_v3.ipynb`
- Données d'origine : `data/social_media.csv` (chargées par le notebook)
- Fichiers produits : `data/final_v3.csv` (jeu de données transformé) et `data/boxplot_ER_reparition_v2.png` (visuel)

---

## 1. Imports
But : importer les bibliothèques utilisées pour la transformation et l'analyse.

- Bibliothèques utilisées : `pandas`, `matplotlib.pyplot`, `seaborn`, `textblob`, `sklearn.preprocessing.LabelEncoder`, `joblib`, `numpy`.
- Note : `TextBlob` est utilisé pour l'analyse de sentiment. Il peut être lent pour de grandes quantités de texte et nécessite des dépendances (par ex. corpora). Vérifier que `pip install textblob` et le téléchargement des ressources nécessaires ont été faits.

---

## 2. Chargement des données
Code : `df = pd.read_csv('../../data/social_media.csv')`

- Entrée : fichier CSV brut.
- Sortie : DataFrame `df`.
- Points d'attention : encodage, valeurs manquantes, formats de date.

---

## 3. Filtrage de la plateforme
Code :
```python
df = df[df['platform'] == 'Instagram']
df.drop(columns=['platform'] , inplace=True)
```

- But : ne conserver que les enregistrements liés à Instagram.
- Entrée : colonne `platform` dans le CSV.
- Sortie : DataFrame réduite sans la colonne `platform`.
- Remarque : suppose que la colonne `platform` contient exactement la chaîne `'Instagram'` (sensible à la casse). Normaliser si nécessaire.

---

## 4. Calcul du Taux d'Engagement (ER)
Code : `df['ER'] = (df['comments_count'] + df['likes'] + df['shares']+ df['views']) / df['follower_count']`

- Formule : ER = (comments_count + likes + shares + views) / follower_count
- Sortie : nouvelle colonne `ER` (valeur fractionnaire). Dans un usage fréquent on multiplie par 100 pour avoir un pourcentage, mais ici le code garde la fraction.
- Points critiques : division par zéro ou `NaN` si `follower_count` est 0 ou manquant — il faut gérer ces cas (filtrer, imputer ou ajouter une petite constante).

---

## 5. Suppression des ER manifestement erronés (> 1)
Code : `df.drop(df[df['ER'] > 1].index , inplace=True)`

- But : enlever les lignes ayant un ER supérieur à 1 (100%) — interprété comme erreurs ou cas extrêmes.
- Remarque : c'est une heuristique. Des posts viraux peuvent dépasser 100% si follower_count est très faible ; décider du seuil selon le contexte métier.

---

## 6. Filtrage des outliers avec mean ± 3*std
Code :
```python
mean = df['ER'].mean()
std = df['ER'].std()
high_limit = mean + 3 * std
low_limit = mean - 3 * std

df = df[df['ER'] <= high_limit]
```

- But : réduire l'effet des valeurs extrêmes en supprimant les ER au-dessus de mean + 3*std.
- Remarque : le code ne retire que la borne supérieure. La borne inférieure `low_limit` est calculée mais non utilisée.
- Impact : si la distribution est très asymétrique, cette méthode garde la majorité des données tout en supprimant les extrêmes de droite.

---

## 7. Statistiques et boxplot de l'ER
Code :
```python
df['ER'].describe()
plt.figure(figsize=(6,4))
sns.boxplot(x=df['ER']/100)
plt.title("Boxplot du Taux d'Engagement (ER)")
plt.savefig('../../data/boxplot_ER_reparition_v2.png' ,  dpi=300, bbox_inches='tight')
plt.show()
```

- But : inspection visuelle et statistique de la distribution de l'ER.
- Remarque : la division par 100 pour la visualisation suppose que l'auteur voulait un format en pourcentage dans le graphique. Le fichier image est stocké dans `data/`.

---

## 8. Extraction des variables temporelles
Code :
```python
df['day'] = pd.to_datetime(df['post_date']).dt.day_name()
df['hour'] = pd.to_datetime(df['post_date']).dt.hour
```

- But : extraire le nom du jour et l'heure à partir de `post_date`.
- Sortie : colonnes `day` (ex. 'Monday') et `hour` (0–23).
- Points d'attention : gérer les dates non parseables (NaT) ; confirmer le fuseau horaire si pertinent.

---

## 9. Score de performance de la description (`description_performance_score`)
Fonction :
- Entrée : chaîne `description` (contenu textuel).
- Sortie : score numérique entre 0 et 1 (arrondi à 3 décimales).

Détail de la logique :
- Normalisation : texte en minuscule, découpe en mots.
- sentiment = TextBlob(text).sentiment.polarity (valeur dans [-1,1])
- word_count = nombre de mots.
- length_score = min(word_count / 30, 1.0) (taille optimale ~30 mots)
- cta_words = ["like","comment","share","follow","dis","pense"]
  - cta_count = nombre d'occurrences exactement égales à ces mots
  - cta_score = min(cta_count / 2, 1.0) (objectif : détecter appels à l'action)
- avg_word_length = longueur moyenne des mots ; readability_score = 1 - min(avg_word_length / 10, 1)

Score final (pondéré) :
- 0.3 * sentiment
- + 0.25 * length_score
- + 0.25 * cta_score
- + 0.2 * readability_score

Retour : valeur arrondie à 3 décimales.

Remarques :
- Le score mélange sentiment (qui peut être négatif) et autres métriques positives — la présence d'un sentiment négatif peut réduire le score.
- Le choix des poids et des seuils (30 mots, cta_count/2) est heuristique ; à valider expérimentalement.

---

## 10. Extraction et évaluation des hashtags
Fonctions principales :
- `extract_hashtags(hashtags: str) -> list[str]` :
  - Si `hashtags` est NaN ou pas une chaîne, retourne []
  - Sépare sur `,`, strip, lowercase, filtre vides
- `count_hashtags(hashtags)` : longueur de la liste extraite

Scoring des hashtags (plusieurs sous-scores) :
1) `hashtag_quantity_score(n: int) -> float` (0..1)
   - n == 0 -> 0.0
   - 5 <= n <= 15 -> 1.0
   - n < 5 -> n / 5
   - n > 15 -> max(0.0, 1 - (n - 15) / 15)  (dégradation progressive au-delà de 15)

2) `hashtag_relevance_score(description, hashtags) -> float`
   - tokenise la description en mots (regex r"\b\w+\b") et compare aux hashtags normalisés
   - score = |desc_words ∩ tag_words| / |tag_words| (proportion de hashtags présents dans la description)

3) `hashtag_diversity_score(hashtags) -> float`
   - calcule la longueur moyenne des hashtags ; retourne min(avg_len / 15, 1.0)
   - idée : hashtags plus longs => potentiellement plus spécifiques

4) `hashtag_specificity_score(hashtags) -> float`
   - proportion de hashtags ayant longueur >= 8

Score global :
- `hashtag_quality_score = 0.30*quantity + 0.30*relevance + 0.20*diversity + 0.20*specificity`
- Retour arrondi à 3 décimales.

Fonction `extract_hashtag_features(row)` : renvoie une Series avec `hashtag_count` et `hashtag_quality_score` pour une ligne.

Remarques pratiques :
- Le séparateur attendu est `,` ; si les hashtags sont fournis avec un autre séparateur (espace, `#` attaché), il faudra adapter.
- `hashtag_relevance_score` ne normalise pas les mots de la description (exclusion des stopwords, lemmatization), c'est une comparaison directe de tokens.

---

## 11. Application des fonctions sur le DataFrame
Code :
```python
df['description_score'] = df['content_description'].apply(description_performance_score)
df[['hashtag_count', 'hashtag_quality_score']] = df.apply(extract_hashtag_features, axis=1)
```

- But : créer des features numériques issues du texte et des hashtags.
- Sortie : colonnes `description_score`, `hashtag_count`, `hashtag_quality_score`.

---

## 12. Encodage cyclique du jour de la semaine
Code :
```python
day_map = { 'Monday':0, ..., 'Sunday':6 }

df['day_num'] = df['day'].map(day_map)
df['day_sin'] = np.sin(2 * np.pi * df['day_num'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_num'] / 7)
```

- But : préserver la nature cyclique des jours (Lundi proche de Dimanche) en encodant via sin/cos.
- Formules :
  - day_sin = sin(2π * day_num / 7)
  - day_cos = cos(2π * day_num / 7)
- Remarque : `day_num` est ensuite supprimé plus bas ; ne pas oublier que sin/cos conservent l'information cyclique.

---

## 13. Encodage des types de contenu et conversion des booléens
Code :
```python
df = pd.get_dummies(df, columns=['content_type'], drop_first=True)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
```

- But : transformer la variable catégorielle `content_type` en colonnes binaires (one-hot) et s'assurer que les colonnes booléennes soient dtype int pour la compatibilité modèles.
- `drop_first=True` évite la multicolinéarité en supprimant la colonne de référence.

---

## 14. Suppression des colonnes brutes
Code :
```python
df.drop(columns=['post_date' , 'content_description' , 'likes' ,'views','shares' ,'hashtags','day_num' ,'day' , 'comments_count'] , inplace=True)
```

- But : retirer les colonnes jugées redondantes ou non nécessaires au modèle final. Les informations utiles ont été extraites sous forme de nouvelles features (ER, description_score, hashtag_* , day_sin/day_cos, hour, etc.).

- Liste des colonnes supprimées :
  - `post_date` (remplacé par `day` et `hour` puis encodage cyclique)
  - `content_description` (remplacé par `description_score`)
  - `likes`, `views`, `shares`, `comments_count` (agrégés dans `ER`)
  - `hashtags` (remplacé par `hashtag_count` et `hashtag_quality_score`)
  - `day_num`, `day` (après création de `day_sin`/`day_cos`)

Remarque : la suppression est définitive dans le DataFrame courant ; si vous voulez garder les colonnes sources pour debug, faire une copie avant.

---

## 15. Suppression des valeurs manquantes et sauvegarde
Code :
```python
df.dropna(inplace=True)
print("Rows with NaN values have been dropped.")
df.to_csv('../../data/final_v3.csv' , index=False)
```

- But : enlever les lignes contenant encore des `NaN` et persister le jeu de données transformé.
- Remarques :
  - `dropna()` peut supprimer de nombreuses lignes si plusieurs champs contiennent des NaN. Il est parfois préférable d'imputer ou d'examiner les colonnes responsables.
  - Fichier de sortie : `data/final_v3.csv` (chemin relatif depuis le notebook)

---

## Points d'attention généraux et recommandations
1. Gestion des zéros et NaN dans `follower_count` avant calcul d'ER : ajouter une imputation ou filtrer les cas 0 pour éviter divisions invalides.
2. Vérifier le séparateur et le format des hashtags ; adapter `extract_hashtags` si nécessaire (espaces, '#' inclus, autres séparateurs).
3. Vérifier la langue du `content_description` : `TextBlob` a des performances variables selon la langue.
4. Documenter/valider les poids heuristiques (dans `description_performance_score` et `hashtag_quality_score`) via analyse et tests empiriques.
5. Avant `dropna()`, lister les colonnes ayant le plus de NaN pour décider d'une stratégie d'imputation plutôt que suppression aveugle.
6. Conserver une version intermédiaire (ex: `df.to_pickle`) avant les suppressions destructrices pour faciliter le debug.

---

## Colonnes finales attendues (exemples)
Après les transformations, le DataFrame final contiendra typiquement :
- `follower_count`, `ER` (target/label), `hour`, `description_score`, `hashtag_count`, `hashtag_quality_score`, `day_sin`, `day_cos`, des colonnes `content_type_*` encodées, et éventuellement d'autres indicateurs bool/entier présents dans le dataset d'origine.

