---
description: Train and compare the readmission models
---
From the clinical_risk_ai/ directory run `python -m src.models.train`.
Report each candidate's ROC-AUC (Logistic Regression, Random Forest,
Gradient Boosting), which model was selected, the saved model path
(models/model.joblib), and the new registry version. Flag any warnings
or unusually high AUC that could indicate leakage.
