#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np;
import pandas as pd;


# In[2]:


df=pd.read_csv("C:\TALENT BATTLE\Machine Learning Project\Week3\stress.csv")


# In[3]:


# df.head(12)


# In[4]:


# Dataframe.describe()


# In[5]:


# Dataframe.isnull().sum()


# In[6]:


import nltk


# In[7]:


import re


# In[8]:


from nltk.corpus import stopwords


# In[9]:


import string


# In[10]:


nltk.download('stopwords')


# In[11]:


stemmer=nltk.SnowballStemmer("english")


# In[12]:


stopword=set(stopwords.words('english'))


# In[13]:


def clean(text):
    text = str(text) . lower()
    text = re. sub('\[.*?\]',' ',text)
    text = re. sub('https?://\S+/www\. \S+', ' ', text)
    text = re. sub('<. *?>+', ' ', text)
    text = re. sub(' [%s]' % re. escape(string. punctuation), ' ', text)
    text = re. sub(' \n',' ', text)
    text = re. sub(' \w*\d\w*' ,' ', text)
    text = [word for word in text. split(' ') if word not in stopword]
    text =" ". join(text)
    text = [stemmer . stem(word) for word in text. split(' ') ]
    text = " ". join(text)
    return text


# In[14]:


df ["text"] = df["text"]. apply(clean)


# In[15]:


import matplotlib. pyplot as plt


# In[16]:


from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[17]:


text = " ". join(i for i in df. text)


# In[18]:


stopwords = set (STOPWORDS)


# In[19]:


wordcloud = WordCloud( stopwords=stopwords,background_color="white") . generate(text)


# In[20]:


plt. figure(figsize=(10, 10) )


# In[21]:


plt. imshow(wordcloud )


# In[22]:


plt. axis("off")


# In[23]:


plt. show( )


# In[24]:


from sklearn. feature_extraction. text import CountVectorizer


# In[25]:


from sklearn. model_selection import train_test_split


# In[26]:


x = np.array (df["text"])
y = np.array (df["label"])


# In[27]:


cv = CountVectorizer ()
X = cv. fit_transform(x)


# In[28]:


print(X)


# In[29]:


xtrain, xtest, ytrain, ytest = train_test_split(X, y,test_size=0.33)


# In[30]:


from sklearn.naive_bayes import BernoulliNB


# In[31]:


model=BernoulliNB()


# In[32]:


model.fit(xtrain,ytrain)


# In[45]:


user=input("Enter the text")
data=cv.transform([user]).toarray()
print(model.predict(data))

