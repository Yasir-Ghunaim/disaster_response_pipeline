import sys
import nltk
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """Load database and split into features and outputs"""
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponse", engine)
    X = df["message"]
    Y = df.drop(columns=["id", "message", "original", "genre"])
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Clean text, apply word tokenization, remove stop words and reduce tokens into their roots"""
    
    # Tokenize text
    text = re.sub(r"[^a-z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    # Reduce tokens into their roots
    stemer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = stemer.stem(tok).strip()
        clean_tok = lemmatizer.lemmatize(clean_tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Return a mechine learning pipeline optimized by GridSearch to classify multi-output dataset """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier(random_state=42)))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__max_depth': [3, 6]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=50)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Print a classification report containing recall, precision and f1-score for each predicted category """
    Y_pred = model.predict(X_test)

    for i in list(range(Y_pred.shape[1])):
        print("Report for column: \"" + category_names[i] + "\"")
        print(classification_report(Y_test.values[:,i], Y_pred[:,i]))
        print("\n\n")


def save_model(model, model_filepath):
    """ Save the model as a pickle file """
    pkl_filename = "disaster_response_model.pkl"  
    with open(pkl_filename, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()