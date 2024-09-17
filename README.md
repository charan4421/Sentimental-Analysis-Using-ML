 Sentiment Analysis on Twitter Data

 This project performs sentiment analysis on a large dataset of tweets (approximately 16 million rows). The goal is to classify the sentiment of the tweets as either positive (1) or negative (0) using various natural language processing (NLP) techniques and machine learning algorithms. The project leverages logistic regression as the machine learning model and follows a structured process for data collection, cleaning, vectorization, and training.
### TOOLS USED PYTHON SC_LEARN, PANDAS , nltp,numpy, deep learning, machine learning
 Table of Contents
 1) [Introduction](#introduction)
 2) [Project Workflow](#project-workflow)
 3) [Technologies Used](#technologies-used)
 4) [Data Collection](#data-collection)
 5) [Data Preprocessing](#data-preprocessing)
 6) [Stopwords Removal](#stopwords-removal)
 7) [Data Cleaning](#data-cleaning)
 8) [Text Stemming](#text-stemming)
 9) [Feature Engineering](#feature-engineering)
 10) [Vectorization](#vectorization)
 11)[Modeling](#modeling)
 12) [Train-Test Split](#train-test-split)
 13) [Logistic Regression](#logistic-regression)
 14) [Evaluation](#evaluation)
 15)  [Model Saving & Reloading](#model-saving--reloading)
 16)   [Conclusion](#conclusion)



workflow of the project for better understanding
 Project Workflow

1. *Data Collection*: Gather tweets using the Twitter API and save them in a CSV file.
2. *Data Preprocessing*: Clean and prepare the text data for analysis.
3. *Feature Engineering*: Convert text data into numerical format using vectorization.
4. *Modeling*: Apply machine learning techniques to classify the sentiment of tweets.
5. *Evaluation*: Measure the accuracy of the model on unseen data.
6. *Deployment*: Save the trained model for future use and reload it when needed.


STEP 1 :
 ### Data Collection
The first step involved collecting tweets using the Twitter API. We extracted data in a zip format and uploaded it to Google Colab, where we unzipped the file and loaded it into a Pandas DataFrame for further processing.

STEP 2 : 
#### Data Preprocessing

### Stopwords Removal
Since the dataset contains around 16 million rows, we began by removing common stopwords (e.g., "and", "the", "is") to focus on more meaningful words.


step 4: 
### Data Cleaning
The text data was cleaned using Python, where we removed unnecessary characters such as punctuation, numbers, and special symbols to simplify the content.
Step 5 :
### Text Stemming
We used the *Porter Stemmer* from NLTK to stem the text, which reduces words to their root form (e.g., "running" becomes "run"). This step helps normalize the data and reduces dimensionality.

python
from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Example of applying stemming to text
df['stemmed_content'] = df['text_column'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))

STEP 6:



## Feature Engineering

### Vectorization
We used a vectorizer (such as *CountVectorizer* or *TF-IDF Vectorizer*) to convert the cleaned and stemmed text data into binary form. This transformation allows the machine learning model to work with numerical data.

python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['stemmed_content'])

STEP 7




## Modeling

### Train-Test Split
We split the data into training and testing sets, with 80% of the data used for training and 20% for testing. The sentiment labels (y) are the target column, containing only 0 (negative) and 1 (positive) values.

python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)




STEP 8




### Logistic Regression
We chose *Logistic Regression* as the machine learning algorithm due to its efficiency and suitability for binary classification tasks.

python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

STEP 9



## Evaluation

After training the model, we evaluated its performance using accuracy metrics on the test set. This helps us understand how well the model generalizes to new, unseen data.

python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

STEP 10


## Model Saving & Reloading

To make our model reusable, we saved it using the pickle module. This allows us to reload the model and use it for future predictions without retraining.

python
import pickle

# Saving the model
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Reloading the model
with open('sentiment_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


## Conclusion

This project demonstrates the application of sentiment analysis on a large dataset of tweets. By following steps from data collection to evaluation, we built a robust model capable of predicting tweet sentiments. The use of logistic regression, coupled with proper preprocessing techniques, resulted in a successful sentiment analysis pipeline.


