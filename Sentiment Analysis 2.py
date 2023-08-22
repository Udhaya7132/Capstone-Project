#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.tokenize.toktok import ToktokTokenizer
import re
tokenizer=ToktokTokenizer()
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner'])


# In[3]:


train=pd.read_csv('C:/Users/UDHAYA KUMAR . R/Desktop/sentiment analysis/archive (1)/Train.csv')


# In[4]:


train.head()


# In[6]:


label_0=train[train['label']==0].sample(n=5000)
label_1=train[train['label']==1].sample(n=5000)


# In[7]:


train=pd.concat([label_1,label_0])
from sklearn.utils import shuffle
train = shuffle(train)


# In[8]:


train.shape


# In[9]:


train.isnull().sum()


# In[10]:


import numpy as np
train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
train.dropna(axis=0, how='any', inplace=True)


# In[12]:


train.head()


# In[13]:


train.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
print('escape seq removed')


# In[14]:


import numpy as np
train.replace(r'^\s*$', np.nan, regex=True,inplace=True)
train.dropna(axis = 0, how = 'any', inplace = True)


# In[15]:


train


# In[16]:


train['text']=train['text'].str.encode('ascii', 'ignore').str.decode('ascii')
print('non-ascii data removed')


# In[17]:


import string
string.punctuation


# In[18]:


def remove_punctuations(text):
    import string
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
train['text']=train['text'].apply(remove_punctuations)


# In[19]:


train


# In[20]:


import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))


# In[21]:


stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


# In[ ]:


stopword_list


# In[ ]:


for i in stopword_list:
    if 'not' and 'no' in i:
        print('yes')
    break        


# In[ ]:


def custom_remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


# In[ ]:


train['text']=train['text'].apply(custom_remove_stopwords)


# In[ ]:


train


# In[ ]:


def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text


# In[ ]:


train['text']=train['text'].apply(remove_special_characters)


# In[ ]:


def remove_html(text):
    import re
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r' ', text)


# In[ ]:


train['text']=train['text'].apply(remove_html)


# In[ ]:


def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r' ',text)


# In[ ]:


train['text']=train['text'].apply(remove_URL)


# In[ ]:


def remove_numbers(text):
    """ Removes integers """
    text = ''.join([i for i in text if not i.isdigit()])         
    return text


# In[ ]:


train['text'] = train['text'].apply(remove_numbers)


# In[ ]:


def cleanse(word):
    rx = re.compile(r'\D*\d')
    if rx.match(word):
        return ''
    return word
def remove_alphanumeric(strings):
    nstrings = [" ".join(filter(None, (
    cleanse(word) for word in string.split()))) 
    for string in strings.split()]
    str1 = ' '.join(nstrings)
    return str1


# In[ ]:


train['text']=train['text'].apply(remove_alphanumeric)


# In[ ]:


train


# In[ ]:


def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# In[ ]:


train['text']=train['text'].apply(lemmatize_text)


# In[ ]:


train['sentiment'] = train['text'].apply(lambda tweet: TextBlob(tweet).sentiment)


# In[ ]:


train


# In[ ]:


sentiment_series = train['sentiment'].tolist()
sentiment_series 


# In[ ]:


columns = ['polarity', 'subjectivity']
df1 = pd.DataFrame(sentiment_series, columns=columns, index=train.index)


# In[ ]:


df1


# In[ ]:


result = pd.concat([train,df1],axis=1)
result


# In[ ]:


result.drop(['sentiment'],axis=1,inplace=True)


# In[ ]:


result


# In[ ]:


result.loc[result['polarity']>=0.3, 'Sentiment'] = "Positive"
result.loc[result['polarity']<0.3, 'Sentiment'] = "Negative"


# In[ ]:


result.loc[result['label']==1, 'Sentiment_label'] = 1
result.loc[result['label']==0, 'Sentiment_label'] = 0


# In[ ]:


result


# In[ ]:




