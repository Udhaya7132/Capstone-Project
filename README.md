# Capstone-Project
Data Analytics Project

Importing Libraries: The necessary libraries such as Pandas, NumPy, TextBlob, NLTK, spaCy, and others are imported.

Data Loading and Preprocessing:
A CSV file named 'Train.csv' is read into a Pandas DataFrame.
The DataFrame is filtered to balance the classes by randomly sampling 5000 samples each for label 0 and label 1.
Rows with empty values are removed.
Special characters, non-ASCII characters, punctuations, stopwords, HTML tags, URLs, and numbers are removed from the 'text' column of the DataFrame.
The text is lemmatized using spaCy.
Sentiment analysis is performed using TextBlob and added to the DataFrame.
A new column 'Sentiment' is generated based on the polarity score, categorizing the sentiment as "Positive" or "Negative".
A new column 'Sentiment_label' is added with labels based on the original 'label' column.

Loading Another Dataset:
A new dataset from the file 'IMDB Dataset.csv' is loaded into a Pandas DataFrame.
Basic exploration and visualization are performed on this dataset.

Text Normalization with NLTK and spaCy:
Text normalization techniques like noise removal, stemming, and stopword removal are applied to the 'review' column of the DataFrame.

Train-Test Split:
The 'review' column is split into training and testing data.

Bag of Words (BoW) and TF-IDF Vectorization:
The CountVectorizer and TfidfVectorizer are used to convert text data into numerical representations for both training and testing data.

Label Encoding:
Sentiment labels are encoded using LabelBinarizer.

Model Training and Prediction:
Logistic Regression models are trained on both BoW and TF-IDF transformed training data.
Predictions are made using these models on the testing data.
Accuracy scores for both models are calculated.



