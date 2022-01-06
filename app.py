import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from operator import itemgetter

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from langdetect import detect
import morfeusz2
import unidecode

class PreprocessingSteamReviews():
    def __init__(self, df_reviews):
        self.df_reviews = df_reviews
        
        self.df_reviews['review'] = self.df_reviews['review'].apply(self.remove_newlines_tabs)
        self.df_reviews['review'] = self.df_reviews['review'].apply(self.strip_html_tags)
        self.df_reviews['review'] = self.df_reviews['review'].apply(self.remove_whitespace)
        self.df_reviews['review'] = self.df_reviews['review'].apply(self.remove_non_alphanumeric_chracters)
        self.df_reviews['review'] = self.df_reviews['review'].apply(self.remove_links)
        self.remove_reviews_with_no_alphanumeric_items()
        self.remove_non_polish_reviews()
        self.lowercase_all()
        self.tokenize_all()
        # self.df_reviews['review'] = self.df_reviews['review'].apply(self.remove_polish_stopwords)
        
        self.morf = morfeusz2.Morfeusz()
        self.df_reviews['review'] = self.df_reviews['review'].apply(self.lemmatisation)
        self.df_reviews['review'] = self.df_reviews['review'].apply(self.remove_polish_stopwords)
        
        self.df_reviews['review'] = self.df_reviews['review'].apply(" ".join)
        
        
    def remove_reviews_under_99_chars(self, n):
        self.df_reviews['len'] = self.df_reviews['review'].str.len()
        self.df_reviews = self.df_reviews[self.df_reviews['len'] > n]
        
    def remove_reviews_under_n_words(self, n):
        self.df_reviews['len2'] = self.df_reviews['review'].str.len()
        self.df_reviews = self.df_reviews[self.df_reviews['len2'] > n]
    
    def remove_newlines_tabs(self, text):
        # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
        Formatted_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ')
        return Formatted_text

    def strip_html_tags(self, text):
        # Initiating BeautifulSoup object soup.
        soup = BeautifulSoup(text, "html.parser")
        # Get all the text other than html tags.
        stripped_text = soup.get_text(separator=" ")
        return stripped_text

    def remove_whitespace(self, text):
        pattern = re.compile(r'\s+') 
        Without_whitespace = re.sub(pattern, ' ', text)
        # There are some instances where there is no space after '?' & ')', 
        # So I am replacing these with one space so that It will not consider two words as one token.
        text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
        return text

    def remove_non_alphanumeric_chracters(self, text):
        regex = re.compile('[^a-zA-ZAaĄąBbCcĆćDdEeĘęFfGgHhIiJjKkLlŁłMmNnŃńOoÓóPpRrSsŚśTtUuWwYyZzŹźŻż ]')
        text = regex.sub('', text)
        return text
    
    def remove_reviews_with_no_alphanumeric_items(self):
        for row, data in self.df_reviews.T.iteritems():
            if not any(c.isalpha() for c in data['review']):
                self.df_reviews.drop([row], inplace=True)
                
    def remove_non_polish_reviews(self):
        for row, data in self.df_reviews.T.iteritems():
            if detect(data['review']) != 'pl':
                self.df_reviews.drop([row], inplace=True)
                
    def lowercase_all(self):
        self.df_reviews['review'] = self.df_reviews['review'].str.lower()
        
    def tokenize_all(self):
        self.df_reviews['review'] = self.df_reviews['review'].str.split()
    
    def remove_polish_stopwords(self, text):
        stopwords = []
        with open("polish.stopwords.txt", encoding = 'utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                stopwords.append(stripped_line)
        words = [word for word in text if word.lower() not in stopwords]
        return words
    
    def lemmatisation(self, text):
        res = []
        for i in text:
            analysis = self.morf.analyse(i)
            x = analysis[0][2][1]
            x = x.split(':')[0].lower()
            res.append(x)
        return res
    
    def remove_links(self, text):
        remove_https = re.sub(r'http\S+', '', text)
        remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
        remove_pl = re.sub(r"\ [A-Za-z]*\.pl", " ", remove_com)
        return remove_pl


app = Flask(__name__)
model = pickle.load(open("best_model.joblib", 'rb'))
vectorizer = pickle.load(open("vectorizer.joblib", 'rb'))
get_features = pickle.load(open("get_features.joblib", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    mess = list(request.form.values())
    df = pd.DataFrame([mess], columns=['review'])
    pre = PreprocessingSteamReviews(df)
    df_pre = pre.df_reviews

    data = vectorizer.transform(df_pre['review'])
    data_t = get_features.transform(data)


    tmp = get_features.get_feature_names_out()
    words = []

    # print(type(data_t.nonzero()[1]))

    for i in data_t.nonzero()[1]:
       words.append([tmp[i], data_t.toarray()[0][i]])
    # print(words)

    data_t = pd.DataFrame(data_t.toarray(), columns=tmp)

    prediction = model.predict(data_t)

    output = ["POZYTYWNA" if prediction == True else "NEGATYWNA"]

    content = {"prediction_text": output[0],
                "important_words": (sorted(words, key=itemgetter(1), reverse=True))}

    return render_template('index.html', **content)

if __name__ == "__main__":
    app.run(debug=True)
