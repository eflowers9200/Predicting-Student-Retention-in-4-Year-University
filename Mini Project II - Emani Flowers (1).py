#!/usr/bin/env python
# coding: utf-8

# In[8]:


#import libraries

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')

from ipywidgets import *
from IPython.display import display

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[9]:


#Load the data
m_data = pd.read_csv('dataset1.csv')
m_data.head()


# In[ ]:


#Explore the Data


# In[10]:


m_data.shape


# In[11]:


m_data.isnull().sum()


# In[12]:


m_data.Target.value_counts()


# In[13]:


m_data.Target.value_counts().plot(kind='bar')
plt.title('Value counts of the Target Variable')
plt.xlabel('Academic Success')
plt.xticks(rotation=0)
plt.ylabel('Count')
plt.show()


# In[26]:


sns.pairplot(m_data)


# In[27]:


corrmat = m_data.corr()
hm = sns.heatmap(corrmat, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 5}, yticklabels = m_data.columns, xticklabels = m_data.columns, cmap = "Spectral_r")
plt.show()


# In[14]:


m_data.columns


# In[15]:


feature_columns = ['Marital status','Displaced', 'Debtor',
       'Tuition fees up to date', 'Gender', 'Scholarship holder',
       'Age at enrollment','Curricular units 1st sem (approved)',
       'Curricular units 2nd sem (approved)', 'Unemployment rate',
       'Inflation rate', 'GDP', 'Target']


# In[36]:


# Copied code from seaborn examples
# https://seaborn.pydata.org/examples/many_pairwise_correlations.html
sns.set(style="white")

# Generate a mask for the upper triangle
mask = np.zeros_like(m_data[feature_columns].corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 18))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(m_data[feature_columns].corr(), mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.show();


# In[38]:


sns.pairplot(m_data[feature_columns])


# In[16]:


def bar_chart(m_data, feature):
    Dropout = m_data[m_data['Target']==1][feature].value_counts(normalize=True)*100
    Graduate = m_data[m_data['Target']==0][feature].value_counts(normalize=True)*100
    df = pd.DataFrame([Dropout,Graduate])
    df.index = ['Dropout','Graduate']
    df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[37]:


# Target Varibale Analysis
bar_chart(m_data, 'Gender')


# In[38]:


m_data['Gender'].value_counts()


# In[39]:


# Target Variable Analysis
bar_chart(m_data, 'Age at enrollment')


# In[40]:


m_data['Age at enrollment'].value_counts()


# In[41]:


# Target Variable Analysis
bar_chart(m_data, 'Scholarship holder')


# In[42]:


m_data['Scholarship holder'].value_counts()


# In[43]:


# Target Variable Analysis
bar_chart(m_data, 'Tuition fees up to date')


# In[44]:


m_data['Tuition fees up to date'].value_counts()


# In[45]:


# Target Variable Analysis
bar_chart(m_data, 'Debtor')


# In[46]:


m_data['Debtor'].value_counts()


# In[47]:


# Target Variable Analysis
bar_chart(m_data, 'Marital status')


# In[49]:


m_data['Marital status'].value_counts()


# In[50]:


# Target Variable Analysis
bar_chart(m_data, 'Displaced')


# In[51]:


m_data['Displaced'].value_counts()


# In[52]:


# Target Variable Analysis
bar_chart(m_data, 'Inflation rate')


# In[54]:


m_data['Inflation rate'].value_counts()


# In[55]:


m_data['Inflation rate'].mean()


# In[56]:


# Target Variable Analysis
bar_chart(m_data, 'GDP')


# In[58]:


m_data['GDP'].value_counts()


# In[59]:


m_data['GDP'].mean()


# In[60]:


# Target Variable Analysis
bar_chart(m_data, 'Unemployment rate')


# In[61]:


m_data['Unemployment rate'].value_counts()


# In[62]:


# Target Variable Analysis
bar_chart(m_data, 'Curricular units 1st sem (approved)')


# In[63]:


m_data['Curricular units 1st sem (approved)'].value_counts()


# In[64]:


# Target Variable Analysis
bar_chart(m_data, 'Curricular units 2nd sem (approved)')


# In[65]:


m_data['Curricular units 2nd sem (approved)'].value_counts()


# In[18]:


feature_cols = ['Marital status','Displaced', 'Debtor',
       'Tuition fees up to date', 'Gender', 'Scholarship holder',
       'Age at enrollment','Curricular units 1st sem (approved)',
       'Curricular units 2nd sem (approved)', 'Unemployment rate',
       'Inflation rate', 'GDP']
X = m_data[feature_cols]
y = m_data['Target']

X
y


# In[19]:


feature_cols


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)


# In[21]:


# Create Model

logreg = LogisticRegression()


# In[22]:


# Fit Model

logreg.fit(X_train, y_train)


# In[23]:


logreg.score(X_train, y_train)


# In[24]:


logreg.score(X_test, y_test)


# In[25]:


# Confusion Matrix
#Evaluate the performance of the model

from sklearn import metrics
y_pred_class = logreg.predict(X_test)
print(metrics.confusion_matrix(y_test, y_pred_class))


# In[26]:


print(np.asarray([['TN', 'FP'], ['FN', 'TP']]))


# In[27]:


con_mat = metrics.confusion_matrix(y_test, y_pred_class)
con_mat


# In[30]:


confusion = pd.DataFrame(con_mat, index=['predicted_graduate','predicted_dropout'],
                         columns=['will_graduate', 'will_dropout'])
confusion


# In[31]:


print(metrics.accuracy_score(y_test, y_pred_class))


# In[34]:


sensitivity = metrics.recall_score(y_test, y_pred_class)
print(sensitivity)


# In[36]:


TP = metrics.confusion_matrix(y_test,y_pred_class)[1,1]
print(TP)
TN = metrics.confusion_matrix(y_test,y_pred_class)[0,0]
print(TN)
FP = metrics.confusion_matrix(y_test,y_pred_class)[0,1]
print(FP)
FN = metrics.confusion_matrix(y_test,y_pred_class)[1,0]
print(FN)

Specificity = TN / (TN + FP)
Specificity


# In[66]:


# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = logreg.predict_proba(X_test)[:,1]

# Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

# Store the Area Under the Curve (AUC) so we can annotate our graph with this metric
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc = "lower right")
plt.show()


# In[ ]:




