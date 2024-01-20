#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score


# ## Loading Data and Data Preprocessing

# In[2]:


pd.set_option('display.max_columns', 100)


# In[3]:


data = pd.read_csv(r"C:\Users\ASUS\Desktop\My_Data_Science_Practise\datasets\german_credit_cleaned.csv")
data


# In[4]:


data.describe(include='all')


# In[5]:


data.columns


# In[6]:


data.dtypes


# In[7]:


data.isnull().sum()


# In[8]:


data['target'].unique()


# In[9]:


data['target'].shape


# In[10]:


data['target'] = data['target'].map({'good':1,'bad':0})


# In[11]:


data.head(5)


# ## Checking Correlation

# In[12]:


data.corr()['target']


# In[13]:


avarage_corr = data.corr()['target'].mean()
avarage_corr


# In[14]:


numeric_columns = data.select_dtypes(include='number')
numeric_columns.columns


# In[15]:


dropped_columns = []

for i in data[['duration', 'loan_amt', 'installment_rate', 'present_residence_since',
       'age', 'num_curr_loans', 'num_people_provide_maint', 'target']]:
    
    if abs(data.corr()['target'][i]) < avarage_corr:
        dropped_columns.append(i)
    
data.drop(dropped_columns, axis=1, inplace=True) 


# In[16]:


data.head(5)


# In[17]:


numeric_columns = data.select_dtypes(include='number')
numeric_columns.columns


# ## Checking Multicollinearity (VIF)

# In[18]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data[[
#     'duration', 
    'loan_amt', 
    'age'
]]

vif=pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif


# In[19]:


data.drop('duration', axis=1, inplace = True)


# In[20]:


data


# ## Checking Outliers

# In[21]:


for i in data[['loan_amt', 'age']]:
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# ## Outlier rule for capping

# In[22]:


Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3-Q1
Lower = Q1-1.5*IQR
Upper = Q3+1.5*IQR


# In[23]:


for i in data[['loan_amt', 'age']]:
    
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i])
    
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[24]:


data = data.reset_index(drop=True)


# In[25]:


data.head(5)


# ## WOE Transformation for Logistic Regression

# In[26]:


new_data = data.copy()


# In[27]:


new_data.head()


# In[28]:


for i in new_data.columns[:-1]:
    if (new_data[i].dtype=='int64') | (new_data[i].dtype=='float64'):
        ranges = [-np.inf, new_data[i].quantile(0.25), new_data[i].quantile(0.5), new_data[i].quantile(0.75), np.inf]
        new_data[i+'category'] = pd.cut(new_data[i], bins=ranges)
        
    if new_data[i].dtype=='object':
        grouped = new_data.groupby([i,'target'])['target'].count().unstack().reset_index()
    else:
        grouped = new_data.groupby([i+'category','target'])['target'].count().unstack().reset_index()
    
    grouped['positive_prop'] = grouped[0] / grouped[0].sum()
    grouped['negative_prop'] = grouped[1] / grouped[1].sum()
    grouped['woe'] = np.log(grouped['positive_prop'] / grouped['negative_prop'])
    grouped.rename(columns={'woe':i+'_woe_quantile'}, inplace=True)
    
    if (new_data[i].dtype=='int64') | (new_data[i].dtype=='float64'):
        new_data = new_data.merge(grouped[[i+'category',i+'_woe_quantile']], how='left', on=i+'category')
        new_data.drop([i+'category',i], axis=1, inplace=True)
    else:
        new_data = new_data.merge(grouped[[i,i+'_woe_quantile']], how='left', on=i)
        new_data.drop(i, axis=1, inplace=True)


# In[29]:


new_data


# ## Splitting Independend and Dependend columns

# In[30]:


X = new_data.drop('target', axis=1)
y = new_data['target']


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Function for model evaluation based on metrics

# In[32]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test = roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    
    accuracy_score_test = metrics.accuracy_score(y_test, y_pred_test)
    accuracy_score_train = metrics.accuracy_score(y_train, y_pred_train)
    
    print('Model Performance:')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Accuracy Score for Test:', accuracy_score_test*100)
    
    print('Accuracy Score for Train:', accuracy_score_train*100)
    
    print('Confusion Matrix:', confusion_matrix)


# ## Modeling for Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression


# In[34]:


lr = LogisticRegression()


# In[35]:


lr.fit(X_train, y_train)


# In[36]:


result_lr = evaluate(lr, X_test, y_test)


# In[37]:


y_prob = lr.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Univariate Analysis

# In[38]:


variables= []
train_Gini=[]
test_Gini=[]

for i in X_train.columns:
    X_train_single=X_train[[i]]
    X_test_single=X_test[[i]]
    
    lr.fit(X_train_single, y_train)
    y_prob_train_single=lr.predict_proba(X_train_single)[:, 1]
    
    
    roc_prob_train=roc_auc_score(y_train, y_prob_train_single)
    gini_prob_train=2*roc_prob_train-1
    
    
    lr.fit(X_test_single, y_test)
    y_prob_test_single=lr.predict_proba(X_test_single)[:, 1]
    
    
    roc_prob_test=roc_auc_score(y_test, y_prob_test_single)
    gini_prob_test=2*roc_prob_test-1
    
    
    variables.append(i)
    train_Gini.append(gini_prob_train)
    test_Gini.append(gini_prob_test)
    

df = pd.DataFrame({'Variable': variables, 'Train Gini': train_Gini, 'Test Gini': test_Gini})

df= df.sort_values(by='Test Gini', ascending=False)

df   


# ## Modeling for other ML algorithms

# In[39]:


data.head()


# ### I will use my first cleaned and preprocessed data for the rest of ML algorithms not WOE-converted data. But, here I will change my data from object to numeric using pd.get_dummies(drop_first=True) and then I will scale it with Standard Scaler.

# In[40]:


data_with_dummy = pd.get_dummies(data, drop_first=True)


# In[41]:


data_with_dummy


# In[42]:


target = data_with_dummy.drop('target', axis=1)
scaled = StandardScaler().fit_transform(target)


# In[43]:


scaled = pd.DataFrame(scaled, columns=target.columns)


# In[44]:


scaled


# In[45]:


scaled['target']= data['target']


# In[46]:


scaled


# In[47]:


X = scaled.drop('target', axis=1)
y = scaled['target']


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[49]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test = roc_score_test*2-1
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred_test)
    
    accuracy_score_test = metrics.accuracy_score(y_test, y_pred_test)
    accuracy_score_train = metrics.accuracy_score(y_train, y_pred_train)
    
    print('Model Performance:')

    print('Gini Score for Test:', gini_score_test*100)
    
    print('Gini Score for Train:', gini_score_train*100)
    
    print('Accuracy Score for Test:', accuracy_score_test*100)
    
    print('Accuracy Score for Train:', accuracy_score_train*100)
    
    print('Confusion Matrix:', confusion_matrix)


# In[50]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


dtc = DecisionTreeClassifier()


# In[52]:


dtc.fit(X_train, y_train)


# In[53]:


result_dtc = evaluate(dtc, X_test, y_test)


# In[54]:


y_prob = dtc.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ### Decision Tree Classification model I used above overfitted since the results show it gave high accuracy and gini score in train data but, did not generalize in unseen (test) data. So, it did not learn well and just memorized patterns. 
# ### Decision trees are capable of learning intricate details of the training data, including noise and outliers. This high flexibility makes them prone to capturing noise rather than the underlying patterns. As a result, the tree may perform exceptionally well on the training data but fail to generalize to new, unseen data.

# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[56]:


rfc = RandomForestClassifier()


# In[57]:


rfc.fit(X_train, y_train)


# In[58]:


result_rfc = evaluate(rfc, X_test, y_test)


# In[59]:


y_prob = rfc.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Applying SelectFromModel

# In[60]:


from sklearn.feature_selection import SelectFromModel


# In[61]:


sfm = SelectFromModel(rfc)
sfm.fit(X_train, y_train)


# In[62]:


selected_feature = X.columns[(sfm.get_support())]
selected_feature


# In[63]:


feature_scores = pd.Series(rfc.feature_importances_, index=X.columns).sort_values(ascending=False)

feature_scores


# In[64]:


X_train=X_train[['loan_amt', 'age', 'checking_acc_status_below_0',
       'checking_acc_status_no_cheking_acc',
       'cred_hist_risky_acc_or_curr_loan_other', 'purpose_car_new',
       'saving_acc_bonds_below_100', 'present_employment_since_below_4y',
       'personal_stat_gender_male:single', 'property_real_estate',
       'other_installment_plans_none', 'telephone_yes']]
X_test=X_test[['loan_amt', 'age', 'checking_acc_status_below_0',
       'checking_acc_status_no_cheking_acc',
       'cred_hist_risky_acc_or_curr_loan_other', 'purpose_car_new',
       'saving_acc_bonds_below_100', 'present_employment_since_below_4y',
       'personal_stat_gender_male:single', 'property_real_estate',
       'other_installment_plans_none', 'telephone_yes']]


# In[65]:


X_train.head()


# In[66]:


X_test.head()


# In[67]:


rfc_importance = RandomForestClassifier()
rfc_importance.fit(X_train, y_train)


# In[68]:


result_rfc_importance = evaluate(rfc_importance, X_test, y_test)


# In[69]:


y_prob = rfc_importance.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Hyperparameter-Tuning

# In[70]:


from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[71]:


rfc_randomized = RandomizedSearchCV(estimator = rfc_importance, param_distributions = random_grid, 
                                    n_iter = 10, 
                                    cv = 5, 
                                    verbose=1, 
                                    random_state=42, 
                                    n_jobs = -1)

rfc_randomized.fit(X_train, y_train)


# In[72]:


result_rfc_randomized = evaluate(rfc_randomized, X_test, y_test)


# In[73]:


y_prob = rfc_randomized.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[74]:


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from mlxtend.classifier import StackingCVClassifier


# In[75]:


xgboost_base = XGBClassifier()


# In[76]:


xgboost_base.fit(X_train, y_train)


# In[77]:


result_xgboost_base = evaluate(xgboost_base, X_test, y_test)


# In[78]:


y_prob = xgboost_base.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[79]:


catboost_base = CatBoostClassifier()


# In[80]:


catboost_base.fit(X_train, y_train)


# In[81]:


result_catboost_base = evaluate(catboost_base, X_test, y_test)


# In[82]:


y_prob = catboost_base.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[83]:


#Hyperparameter Tuning (XGBoost)

param_distributions = {
    
    'n_estimators': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7, 10],
    'subsample': np.linspace(0.5, 1, num=6),
    'colsample_bytree': np.linspace(0.5, 1, num=6),
    'gamma': [0,1,5,10]
    
}

param_distributions


# In[84]:


xgboost_randomized = RandomizedSearchCV(xgboost_base, 
                                        param_distributions=param_distributions, 
                                        n_iter=10, cv=5, 
                                        n_jobs=-1, 
                                        random_state=42)
xgboost_randomized.fit(X_train, y_train)


# In[85]:


optimized_xgboost=xgboost_randomized.best_estimator_


# In[86]:


result_optimized_xgboost=evaluate(optimized_xgboost, X_test, y_test)


# In[87]:


y_prob = optimized_xgboost.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[88]:


#Hyperparameter Tuning (CatBoost)

param_distributions = {
    
    'iterations': [10, 50, 100, 500, 1000],
    'learning_rate': [0.01, 0.1, 1.0],
    'depth': [3, 5, 7, 9],
    'l2_leaf_reg': np.linspace(2, 30, num=7)
    
}

param_distributions


# In[89]:


catboost_randomized=RandomizedSearchCV(catboost_base, 
                                       param_distributions=param_distributions, 
                                       cv=5, n_iter=10, 
                                       random_state=42)

catboost_randomized.fit(X_train, y_train)


# In[90]:


optimized_catboost=catboost_randomized.best_estimator_


# In[91]:


result_optimized_catboost=evaluate(optimized_catboost, X_test, y_test)


# In[92]:


y_prob = optimized_catboost.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# In[93]:


#Stacking Model

base_classifiers = [
    catboost_base,
    optimized_xgboost,
    rfc
]


# In[94]:


meta_classifier = optimized_catboost


# In[95]:


stacking_classifier = StackingCVClassifier(classifiers=base_classifiers,
                                           meta_classifier=meta_classifier,
                                           cv=5,
                                           use_probas=True,
                                           use_features_in_secondary=True,
                                           verbose=1,
                                           random_state=42)


# In[96]:


stacking_classifier.fit(X_train, y_train)


# In[97]:


result_stacking_classifier = evaluate(stacking_classifier, X_test, y_test)


# In[98]:


y_prob = stacking_classifier.predict_proba(X_test)[:,1]

roc_score = roc_auc_score(y_test, y_prob)
gini_score = (roc_score*2)-1

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure()

plt.plot(fpr, tpr, label='(Roc_Score = %0.2f)' % roc_score)
plt.plot(fpr, tpr, label='(Gini_Score = %0.2f)' % gini_score)
plt.plot([0, 1], [0, 1])
plt.xlim()
plt.ylim()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend(loc='best')
plt.show()


# ## Conclusion

# ### I applied so many different machine learning algorithms including xgboost, catboost, stacking etc. to dataset, and then optimized them as well. However Logistic Regression with WOE values remained the most accurrate model with high (over 62) gini score. That's why I checked above Univariate Analysis for Logistic Regression model.
