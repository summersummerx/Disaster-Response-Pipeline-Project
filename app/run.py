import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    clean and transform text into tokens
    input: sentences from document
    output: a list of cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def tokenize_self(text):
    '''
    added extra step to remove punctuations from text
    as we'd like to count text length without punctuations
    clean and transform text into tokens
    input: sentences from document
    output: a list of cleaned tokens
    '''
    #remove punctuations
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    #normalization
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
# load data
engine = create_engine('sqlite:////home/workspace/data/DisasterResponse.db', echo=True)
df = pd.read_sql_table("drmessage", engine)
df_lang = pd.read_sql_table("messagelang", engine)

# load model
model = joblib.load("/home/workspace/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # data used for graph1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # data used for graph2
    df['message_length'] = df.message.apply(lambda x:len(tokenize_self(x)))
    genre_mlen = df.groupby('genre')['message_length'].mean()
    genre_mlen_names = list(genre_mlen.index)
    
    # data used for graph3
    lang_counts = df_lang[df_lang.genre=='direct']['counts']
    lang_names = list(df_lang.language)
    
    
    
    # create visuals
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
                Scatter(
                    x=genre_mlen_names,
                    y=genre_mlen,
                    mode = 'markers',
                    marker=dict(
                        color='LightSkyBlue',
                        size=20,
                        line=dict(
                            color='MediumPurple',
                            width=2
                                 )
                       )
                )
            ],

            'layout': {
                'title': 'Average Message Length by Message Genres',
                'yaxis': {
                    'title': "Avg Message Length"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Pie(
                    labels=lang_names,
                    values=lang_counts,
                    textposition='inside'
                )
            ],

            'layout': {
                'title': 'Launguage used in Messages of Direct Genre'
            }
        }

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
