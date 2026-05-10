# 📉 Prédiction du Churn Client — Plateforme Fintech

> **Un client sur le point de partir, peut-on le détecter avant qu'il ne parte ?**  
> Classification supervisée pour anticiper le churn sur une plateforme financière digitale.



## 🗂️ Contexte du projet

Dans le secteur fintech, perdre un client coûte bien plus cher que d'en acquérir un nouveau. Ce projet s'attaque à cette problématique en construisant un **modèle de classification** capable de prédire si un utilisateur va quitter la plateforme (`churn = 1`) ou rester (`churn = 0`).

Le dataset contient **27 variables** décrivant le comportement financier et l'activité des utilisateurs : dépôts, retraits, score de crédit, prêts, récompenses, plateformes utilisées, etc.



## 🎯 Questions posées

- Quelles variables sont les plus corrélées au churn ?
- Quel modèle prédit le mieux les départs clients ?
- Comment optimiser les hyperparamètres pour améliorer les performances ?



## 🔍 Démarche analytique

```
Chargement → Nettoyage → Preprocessing → EDA → Modélisation → Optimisation → Sauvegarde
```

| Étape | Action |
|---|---|
| **Exploration** | Shape, dtypes, describe, valeurs manquantes, doublons |
| **Nettoyage** | Remplacement des `'na'` string par `NaN`, suppression des colonnes inutiles (`user`, `zodiac_sign`) |
| **Imputation** | `SimpleImputer` — médiane pour les colonnes numériques, mode pour les catégorielles |
| **Outliers** | Méthode IQR — remplacement par les bornes |
| **Encodage** | `LabelEncoder` pour `housing`, Frequency Encoding pour `payment_type` |
| **Split & Normalisation** | `train_test_split` (80/20, stratifié), `StandardScaler` |
| **Modélisation** | 5 modèles comparés : Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost |
| **Optimisation** | `GridSearchCV` (cv=10) + `cross_val_score` sur Random Forest |
| **Sauvegarde** | Export du meilleur modèle via `pickle` |



## 🤖 Modèles comparés

| Modèle | Type |
|---|---|
| Logistic Regression | Linéaire |
| Decision Tree | Arbre de décision (critère : entropie) |
| Random Forest | Ensemble (bagging) |
| SVM | Support Vector Machine (kernel linéaire) |
| XGBoost | Gradient Boosting |

> **Évaluation** : Confusion matrix, Classification report (precision / recall / f1-score), Courbe ROC



## 💡 Variables clés identifiées

- `credit_score` — Score de crédit, fortement discriminant
- `rewards_earned` / `reward_rate` — Niveau d'engagement client
- `age` — Profil démographique
- `housing` / `payment_type` — Comportement financier
- Indicateurs d'inactivité : `left_for_two_month_plus`, `left_for_one_month`



## 🛠️ Stack technique

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightgrey?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)



## 📁 Structure du repo

```
📦 churn-prediction
 ┣ 📓 LOWE_KENGNE_IABD_B3.ipynb   # Notebook principal
 ┣ 📄 churn_data.csv               # Dataset source
 ┣ 🤖 model.pkl                    # Modèle sauvegardé
 ┗ 📖 README.md
```



## ▶️ Lancer le projet

```bash
git clone https://github.com/lkmc77/churn-prediction.git
cd churn-prediction
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
jupyter notebook LOWE_KENGNE_IABD_B3.ipynb
```



## 👤 Auteur

**Murielle Lowe** · Étudiante Bachelor IA & Big Data  
📍 Yaoundé, Cameroun  
