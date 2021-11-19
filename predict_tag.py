import flask
import sklearn as skl
import re
import nltk
import os
import pandas as pd 
import numpy as np 
import pickle
import pandas as pd
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from ast import literal_eval
# from Multi_labelling import y_test_inversed, y_test_predicted_labels_tfidf_rfc

df = pd.read_csv('dataset_clean.csv',converters={"list_title": literal_eval,
                               "list_body": literal_eval,
                               "Tags_list": literal_eval})

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

X = df['list_title'] + df['list_body']
Y = df["Tags_list"]

# Initialize the "CountVectorizer" TFIDF for Full_doc
vectorizer = TfidfVectorizer(analyzer="word",
                             max_df=.6,
                             min_df=0.005,
                             tokenizer=None,
                             preprocessor=' '.join,
                             stop_words=None,
                             lowercase=False)

vectorizer.fit(X)

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(Y)
# X_tfidf = vectorizer.transform(X)


'''
function to clean a list containing the words of the sentence.
- Replace capitals letters
- remove punctuation
- remove digits

arg : 
    t : a list of strings 
'''


def clean_txt(t):
    
    for i in range(len(t)) :
        
        t[i] = t[i].lower()
        t[i] = re.sub(r'[^\w\s]','',t[i])
        t[i] = ''.join([j for j in t[i] if not j.isdigit()])
        
    return t 

def remove_stop_words(t,stop_words = stopwords.words('english')):
    stop_words.extend(['from', 'use','would','know','way','need','seem','example',
                      'want','try','make','give','get','like','one','set','anyone','x',
                      'go','file','change','code','look','create','question','question',
                      'something','possible','nt','add','see','page','work',
                      'service','option','could'])
    filtered_sentence = [w for w in t if not w in stop_words]
    filtered_sentence = list(filter(None,filtered_sentence))
    return filtered_sentence

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

'''
function to lemmatize a list of strings

arg : 
    - sentence : list of str to lemmatize
    - lemmatizer : nltk lemmatizer function to use
'''

def get_lemmatize_sentence(sentence, lemmatizer = WordNetLemmatizer()):
    
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in sentence]

def preprocess(sentence):
    s = sentence.split()
    s = clean_txt(s)
    s = remove_stop_words(s)
    s = get_lemmatize_sentence(s)
    return s

def ValuePredictor(sentence):
 to_predict = preprocess(sentence)

 sentence_tfidf = vectorizer.transform([to_predict])
 
 loaded_model = pickle.load(open('model.pkl','rb'))
 y_test_predicted_labels_tfidf_rfc = loaded_model.predict(sentence_tfidf)

        # Inverse transform
 y_test_pred_inversed_rfc = multilabel_binarizer\
    .inverse_transform(y_test_predicted_labels_tfidf_rfc)
 
 result = y_test_pred_inversed_rfc
 l = len(result)
#  for i in range(l):
#     print(result[i])
 return result


app = Flask(__name__)

@app.route("/")
def home():
   return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def result():
      if request.method == 'POST':
        sentence1 = request.form['title']
        sentence2 = request.form['body']
              
        to_predict_list1 = sentence1 + ' ' + sentence2

#         to_predict_list = preprocess(to_predict_list1)
        prediction = ValuePredictor(to_predict_list1)
#         prediction = str(result)
        return render_template('predict.html',prediction=prediction)
#     sentence1 = 'when I run command php artisan optimize  this error appeares'
#     sentence2 = 'Then I have to go to bootstrap cache directory and need to delete files then error disappear. What is the issue how I   could solve it ?'
#     sentence = sentence1 + ' ' + sentence2
#     res = ValuePredictor(sentence)
#     return f" Votre question : <br><br> {sentence1} <br><br> {sentence2} <br><br> Le(s) Tag(s) détecté(s) pour votre question : {str(res)}"

if __name__ == "__main__":
   app.run("0.0.0.0",5000)