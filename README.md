# Soccer Match Outcome Prediction Using Ensemble Learning

A comparative analysis of ensemble learning methods (Random Forest, XGBoost, Voting Ensemble) for predicting soccer match outcomes as three-class classification: **Home Win**, **Draw**, or **Away Win**.

## Dataset

**European Soccer Database** from Kaggle — 25,000+ matches across 11 European countries (2008–2016).

Download: [kaggle.com/datasets/hugomathien/soccer](https://www.kaggle.com/datasets/hugomathien/soccer)

Place `database.sqlite` in the project root directory before running any scripts.

## Setup

```bash
pip install -r requirements.txt
```

## Project Structure

```
soccer-match-prediction/
├── 01_explore_database.py       # Explore SQLite tables and structure
├── 02_extract_and_preprocess.py # Data extraction and feature engineering
├── 03_train_models.py           # Model training and evaluation (coming soon)
├── requirements.txt
├── .gitignore
└── README.md
```

## Engineered Features

- **Recent form** — rolling results of last 5 matches per team
- **Head-to-head record** — historical win/loss/draw between matchup teams
- **Home advantage** — binary indicator
- **Goal difference trends** — rolling average goal differential
- **Aggregated player ratings** — average FIFA attack/defense/midfield per team
- **Betting odds** — market-implied probabilities

## Models

| Model | Strategy | Key Idea |
|-------|----------|----------|
| Random Forest | Bagging | Independent trees, majority vote |
| XGBoost | Boosting | Sequential error correction |
| Voting Ensemble | Stacking | RF + XGBoost + Logistic Regression combined |

## Evaluation

- Accuracy, Weighted F1, AUC-ROC (One-vs-One)
- Confusion matrices, training time comparison
- SHAP feature importance analysis

## References

- Atta Mills et al. (2024). *Data-driven prediction of soccer outcomes.* Journal of Big Data.
- Carpita et al. (2019). *Exploring and modelling team performances of the Kaggle European Soccer database.* Statistical Modelling.
- Groll et al. (2019). *Hybrid machine learning forecasts for the FIFA Women's World Cup 2019.* arXiv.
- Procedia Computer Science (2025). *Ensemble Methods and Feature Selection for EPL Match Outcome.*
