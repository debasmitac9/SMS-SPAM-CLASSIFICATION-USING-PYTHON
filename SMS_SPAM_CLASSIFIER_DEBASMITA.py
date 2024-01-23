#!/usr/bin/env python
# coding: utf-8

# In[13]:


#importing modules
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
import re
from nltk.corpus import stopwords


# In[14]:


#loading dataset
df = pd.read_csv('SPAM text message 20170820 - Data.csv')
df.tail()
df.head()


# In[15]:


#get necessary columns for processing
df = df[['Message', 'Category']]
# df.rename(columns={'Message': 'sms', 'Category': 'label'}, inplace=True)
df = df.rename(columns={'Message': 'sms', 'Category': 'label'})
df.head()


# In[16]:


#preprocessing the dataset
# check for null values
df.isnull().sum()


# In[17]:


STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    # convert to lowercase
    text = text.lower()
    # remove special characters
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # remove stopwords
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


# In[18]:


# clean the messages
df['clean_text'] = df['sms'].apply(clean_text)
df.head()


# In[19]:


#input split
X = df['clean_text']
y = df['label']


# In[20]:


#model training
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer

def classify(model, X, y):
    # train test split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True, stratify=y)
    # model training
    pipeline_model = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', model)])
    pipeline_model.fit(x_train, y_train)
    
    print('Accuracy:', pipeline_model.score(x_test, y_test)*100)
    
#     cv_score = cross_val_score(model, X, y, cv=5)
#     print("CV Score:", np.mean(cv_score)*100)
    y_pred = pipeline_model.predict(x_test)
    print(classification_report(y_test, y_pred))


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model, X, y)


# In[22]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
classify(model, X, y)


# In[23]:


from sklearn.svm import SVC
model = SVC(C=3)
classify(model, X, y)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model, X, y)


# In[ ]:





# In[ ]:





# In[ ]:




