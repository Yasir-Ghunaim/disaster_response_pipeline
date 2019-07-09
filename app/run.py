import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/disaster_response_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # get most 10 frequent categories
    categoryDf = df.drop(columns=["id", "message", "original", "genre"])
    categoryDf_counts = ((categoryDf.sum()/categoryDf.shape[0]).sort_values(ascending=False))[:10]

    # calculate average used categories for each genre
    genreDf = df.drop(columns=["id", "message", "original"])
    genreDf_count = genreDf.groupby("genre").sum().sum(axis=1).astype(float)
    for i in range(genreDf_count.index.shape[0]):
        genreLength = genreDf[genreDf["genre"] == genreDf_count.index[i]].shape[0]
        genreDf_count[i] = genreDf_count[i]/genreLength
    genreDf_count = genreDf_count.sort_values(ascending=False)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categoryDf_counts.index,
                    y=categoryDf_counts.values
                )
            ],

            'layout': {
                'title': 'Most Frequent Categories',
                'yaxis': {
                    'title': "Percent (%)"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genreDf_count.index,
                    y=genreDf_count.values
                )
            ],

            'layout': {
                'title': 'Average Number of Used Categories per Genre',
                'yaxis': {
                    'title': "Average"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()