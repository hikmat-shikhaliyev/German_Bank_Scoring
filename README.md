# German_Bank_Scoring
## Overview
This GitHub repository contains a comprehensive German Credit Scoring project focusing on predicting the creditworthiness of individuals. The project covers various stages, including data loading, preprocessing, exploratory data analysis (EDA), feature engineering, and the implementation of multiple machine learning models.
# Project Structure
### 1. Importing Libraries
The project begins by importing essential libraries for data manipulation, visualization, and machine learning, including Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

### 2. Loading Data and Data Preprocessing
The code then loads the German Credit dataset from a CSV file using the pd.read_csv() function. It sets the option to display all columns using pd.set_option('display.max_columns', 100). The dataset is then described using data.describe(include='all'), and the column names and data types are displayed using data.columns and data.dtypes, respectively. The code checks for missing values using data.isnull().sum() and displays the unique values in the 'target' column using data['target'].unique(). The 'target' column is then mapped to binary values using data['target'] = data['target'].map({'good':1,'bad':0}). Initial data exploration involves checking column information, data types, and handling missing values.
### 3. Checking Correlation
Correlation analysis is performed to identify features strongly correlated with the target variable. The average correlation threshold is calculated, and features with correlations below this threshold are dropped.

### 4. Checking Multicollinearity (VIF)
Variance Inflation Factor (VIF) is calculated to identify and handle multicollinearity among numeric features. Features with high VIF values (greater than 5) are removed.

### 5. Checking Outliers
Outliers in numeric features (loan amount and age) are visualized using box plots. An outlier capping rule is applied to mitigate the impact of outliers.

### 6. WOE Transformation for Logistic Regression
The Weight of Evidence (WOE) transformation is applied to numeric and non-numeric features for logistic regression. New WOE-transformed features are created to enhance model interpretability.

### 7. Univariate Analysis
Univariate analysis is conducted to evaluate the performance of logistic regression models for individual features. Gini scores are calculated to identify influential features.

### 8. Modeling
Several machine learning algorithms are implemented, including Logistic Regression, Decision Tree Classifier, Random Forest Classifier, XGBoost Classifier, and CatBoost Classifier. The models are evaluated using metrics such as ROC curve, Gini score, and accuracy.

### 9. Hyperparameter Tuning
Hyperparameter tuning is performed for the XGBoost and CatBoost models using randomized search to optimize their performance.

### 10. Stacking Model
A stacking model is created using a combination of base classifiers (CatBoost, XGBoost, and Random Forest) with a meta-classifier (CatBoost). The stacking model's performance is evaluated and compared with individual models.
