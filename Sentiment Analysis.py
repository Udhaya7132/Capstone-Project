#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[10]:


df = pd.read_csv("C:/Users/UDHAYA KUMAR . R/Desktop/sentiment analysis/IMDB Dataset.csv")


# In[11]:


df.head()


# In[12]:


df.columns


# In[13]:


df.isnull().sum()


# In[14]:


print(f'The no of rows: {df.shape[0]}')
print(f'The no of columns:{df.shape[1]}')


# In[15]:


df.describe()


# In[16]:


df.info()


# In[9]:


df['sentiment'].value_counts()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(data=df,x='sentiment')
plt.title("Sentiment Distribution")
plt.show()


# # Text Normalization

# In[18]:


import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize


# In[19]:


get_ipython().system('pip install spacy')
get_ipython().system('pip install textblob')

import spacy
import re,string,unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from textblob import Word
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from bs4 import BeautifulSoup


# In[20]:


nltk.download('stopwords')


# In[19]:


#Tokenization 
token= ToktokTokenizer()
#setting English words
stopwords=nltk.corpus.stopwords.words('english')


# In[21]:


#Removing Noisy Text

def noiseremoval_text(text):
    soup=BeautifulSoup(text,'html.parser')
    text=soup.get_text()
    text=re.sub('\[[^]]*\}]', '', text)
    return text


# In[22]:


#Apply function on review column
df['review']=df['review'].apply(noiseremoval_text)


# In[24]:


df.head()


# # Stemming

# In[33]:


def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text=' '.join([ps.stem(word) for word in text.split()])
    return text


# In[34]:


#Apply Function on review column
df['review']=df['review'].apply(stemmer)


# In[37]:


df.head()


# In[52]:


#Removing the Stopwords

def removing_stopwords(text, is_lower_case=False):
    #Tokenization of the txt
    tokenizers=ToktokTokenizer()
    #setting English Stopwords
    tokens=tokenizers.tokenize(text)
    tokens=[i.strip() for i in tokens]
    
    if is_lower_case:
        filtokens=[i for i in tokens if tokens not in stop_wr]
    else:
        filtokens=[i for i in tokens if i.lower() not in stop_wr]
        
    filtered_texts= ' '.join(filtokens)
    return filtered_texts


# In[53]:


df['review']=df['review'].apply(removing_stopwords)

# Import necessary libraries
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer

# Define the stopwords
stop_wr = set(stopwords.words('english'))

# Define the function to remove stopwords
def removing_stopwords(text, is_lower_case=False):
    # Tokenization of the text
    tokenizers = ToktokTokenizer()
    # Setting English Stopwords
    tokens = tokenizers.tokenize(text)
    tokens = [i.strip() for i in tokens]
    
    if is_lower_case:
        filtokens = [i for i in tokens if tokens not in stop_wr]
    else:
        filtokens = [i for i in tokens if i.lower() not in stop_wr]
        
    filtered_texts = ' '.join(filtokens)
    return filtered_texts

# Apply the function to the 'review' column of the DataFrame
df['review'] = df['review'].apply(removing_stopwords)

# In[55]:


df.head()


# # Train Test Split

# In[63]:


#train Dataset
train_reviews_data=df.review[:30000]


# In[64]:


#test Dataset
test_reviews_data=df.review[30000:]


# # Bag of words

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer
#Count vectorizer 
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1,3))
#transformed train reviews
cv_train = cv.fit_transform(train_reviews_data)
#TEst
cv_test = cv.transform(test_reviews_data)

print(cv_train.shape)
print(cv_test.shape)


# # TF_IDF

# In[67]:


tf=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#Transformed train reviews
tf_train = tf.fit_transform(train_reviews_data)
#Test
tf_test = tf.transform(test_reviews_data)
print(tf_train.shape)
print(tf_test.shape)


# # LabelEncoding

# In[69]:


#labeling the sentiment data
label=LabelBinarizer()
#transformed sentiment data
sentiment_data = label.fit_transform(df['sentiment'])
print(sentiment_data.shape)


# In[70]:


print(sentiment_data)


# In[71]:


train_data = df.sentiment[:30000]
test_data = df.sentiment[30000:]


# In[ ]:


#training the model
logistic = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
#fitting the model for bag of words
lr_bow = logistic.fit(cv_train,train_data)
print(lr_bow)
#fitting the model for tfidf
lr_tf=logistic.fit(tf_train,train_data)
print(lr_tf)


# In[ ]:


#Predicting the model for bag of words
bow_predict = logistic.predict(cv_test)
print(bow_predict)


# In[ ]:


#accuracy score for bow
lr_bow_score = accuracy_score(test_data,bow_predict)
print('lr_bow_score:',lr_bow_score)


# In[ ]:


#Fitting the model for tfidf features

lr_tf = logistic.fit(tf_train,train_data)
print(lr_tf)


# In[ ]:


#predicting the model for tfidf 
lr_tf_predict = logistic.predict(tf_test)
print(lr_tf_predict)


# In[ ]:


#Accuracy score for tfidf 

lr_tf_score = accuracy_score(test_data,lr_tf_predict)
print(lr_tf_score)


# In[ ]:




