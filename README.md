# wine_quality_ML
Machine learning models on wine dataset


* Wine Quality Classification (White Wine Dataset)

This project applies multiple machine learning algorithms to classify white wine quality as **good** or **bad** based on physicochemical properties. The dataset includes chemical measurements such as acidity, sugar, pH, sulphates, and alcohol content.

The goal is to compare different ML models and evaluate their performance using metrics like accuracy, ROC-AUC, confusion matrix, and cross-validation.

---

* Dataset

The project uses the **White Wine Quality Dataset** from the UCI Machine Learning Repository.

Files included in this repository:

- `winequality-white.csv` — original dataset  
- `cleaned_winequality_data.csv` — cleaned dataset (duplicates removed)

The target variable `quality` is converted into a binary label:

- **1 → Good quality** (quality ≥ 6)  
- **0 → Bad quality** (quality < 6)

---

* Models Implemented

The following machine learning models were trained and evaluated:

- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Classifier (SVC)**

Each model includes:

- Hyperparameter tuning using **GridSearchCV**
- Cross-validation performance
- Confusion matrix
- ROC curve
- Classification report

---

* Results Summary

Each model was evaluated using:

- Accuracy Score  
- Classification Report  
- Confusion Matrix  
- ROC Curve  
- Cross-Validation Score  

**Random Forest and SVC achieved the strongest overall performance.**



