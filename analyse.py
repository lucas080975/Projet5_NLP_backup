import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Visualisation de la database

df = pd.read_csv('QueryResultsSO.csv')
df.head()

df.info()

#Récupération des colonnes utiles

df=df[['Id','CreationDate','Score','Body','Title','Tags','ViewCount','AnswerCount','CommentCount']]
df.dropna(subset = ['Tags'],inplace = True)
df.info()

#visualisation des data

df.describe()

#Visualition du score, Views

fig, axs = plt.subplots(2, 2,figsize=(12,12))

df['Score'].plot.hist(ax = axs[0,0])

df['ViewCount'].plot.hist(ax = axs[1,0])

df['AnswerCount'].plot.hist(ax = axs[0,1])

df['CommentCount'].plot.hist(ax = axs[1,1])


axs[0,0].set_title('Score Histogramme')
axs[0,0].set_xlabel('Id')
axs[0,0].set_ylabel('Nb of questions')

axs[1,0].set_title('ViewCount Histogramme')
axs[1,0].set_xlabel('Id')
axs[1,0].set_ylabel('Nb of questions')

axs[0,1].set_title('AnswerCount Histogramme')
axs[0,1].set_xlabel('Id')
axs[0,1].set_ylabel('Nb of questions')

axs[1,1].set_title('CommentCount Histogramme')
axs[1,1].set_xlabel('Id')
axs[1,1].set_ylabel('Nb of questions')

plt.show()

#Visualisation de la longueur des titres des questions

fig = plt.figure(figsize=(15, 12))

ax = sns.countplot(x=df.Title.str.len(),palette = 'crest')
start, end = ax.get_xlim()
ax.xaxis.set_ticks(np.arange(0, end, 5))
plt.axvline(df.Title.str.len().median() - df.Title.str.len().min(),
            color="r", linestyle='-.',
            label="taille médiane du titre : "+str(df.Title.str.len().median()))

ax.set_xlabel("longueur du titre")
ax.set_ylabel('Nb de questions')
plt.title("Longueur du titre des questions",
          fontsize=18, color="black")
plt.legend()
plt.show()


# Visualisation de la longueur du corps des questions 

fig = plt.figure(figsize=(15, 12))
ax = sns.histplot(x=df.Body.str.len(),palette = 'Paired',bins= 100)
start, end = ax.get_xlim()

plt.axvline(df.Body.str.len().median() - df.Body.str.len().min(),
            color="r", linestyle='-.',
            label="taille médiane du titre : "+str(df.Body.str.len().median()))
ax.set_xlabel("longueur du corps")
ax.set_ylabel('Nb de questions')
plt.title("Longueur du corps des questions",
          fontsize=18, color="black")

plt.legend()
plt.show()

#Suppression des lignes avec un corps > 5000 mots

df = df[df.Body.str.len() <= 5000]


### Analyse des tags

df['Tags'].head(7)

# on remplace les '< >'
df['Tags'] = df['Tags'].str.translate(str.maketrans({'<': '', '>': ','}))

# suppression de la dernière "," pour chaque ligne
df['Tags'] = df['Tags'].str[:-1]
df['Tags'].head(8)

# Création d'une liste pour visualiser les tags les plus fréquents 

separator = ','
list_words = []
for word in df['Tags'].str.split(separator):
    list_words.extend(word)
df_list_words = pd.DataFrame(list_words, columns=["Tag"])
df_list_words = df_list_words.groupby("Tag").agg(tag_count=pd.NamedAgg(column="Tag", aggfunc="count"))
df_list_words.sort_values("tag_count", ascending=False, inplace=True)

print("Le jeu de données compte {} tags.".format((df_list_words.shape[0])))

fig = plt.figure(figsize=(15, 8))
sns.barplot(data=df_list_words.iloc[0:50, :],
            x=df_list_words.iloc[0:50, :].index,
            y="tag_count", palette = 'crest')
plt.xticks(rotation=70)
plt.ylabel('Occurence du tag')
plt.title("50 Tags les plus utilisés dans la base de données",
          fontsize=18)
plt.show()

#Nombre de tags par question

df['Tags_list'] = df['Tags'].str.split(',')
df['Tags_count'] = df['Tags_list'].apply(lambda x: len(x))

# Plot the result
fig = plt.figure(figsize=(12, 8))
ax = sns.countplot(x=df.Tags_count,palette = 'Paired')
ax.set_xlabel("Tags")
ax.set_ylabel('Nb de questions')
plt.title("Nombre de tags par question",
          fontsize=18)
plt.show()

# Retrait des tags les moins utilisés

def filter_tag(x, top_list):
    """Comparison of the elements of 2 lists to 
    check if all the tags are found in a list of top tags.

    Parameters
    ----------------------------------------
    x : list
        List of tags to test.
    ----------------------------------------
    """
    temp_list = []
    for item in x:
        if (item in top_list):
            #x.remove(item)
            temp_list.append(item)
    return temp_list

top_tags = list(df_list_words.iloc[0:50].index)
df['Tags_list'] = df['Tags_list']\
                    .apply(lambda x: filter_tag(x, top_tags))
df['number_of_tags'] = df['Tags_list'].apply(lambda x : len(x))
df = df[df.number_of_tags > 0]
print("New size of dataset : {} questions.".format(df.shape[0]))

df.head(10)

# Preprocessing des features

### retrait des balises HTML

#Suppression des balises HTML 
'''
remove_balise_html(html_doc, parser = 'html.parser')
remove html balises on an str file
args : 
    - html_doc = html document to clean
    - parser
'''
from bs4 import BeautifulSoup

def remove_balise_html(html_doc,parser='html.parser'):
    cleantext = BeautifulSoup(html_doc, parser).text
    return cleantext

df['Body'] = df['Body'].apply(remove_balise_html)
df['Title'] = df['Title'].apply(remove_balise_html)

df[['Body','Title']].head()

### Tokenization

#Importation des packages necessaires

import nltk
nltk.download('popular')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize


# Word tokenization sur les colonnes Body et Title

df['list_body'] = df['Body'].apply(word_tokenize)
df['list_title'] = df['Title'].apply(word_tokenize)

df[['list_body','list_title']].head()


df.head()

### Normalisation

    - Mettre tous les mots en minuscules
    - retirer la ponctuation
    - retirer les stopwords
    - retirer d'autres caractères non utiles pour le multi-labelling classification

from nltk.corpus import stopwords
from collections import Counter
import re

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
    

'''
function to clean a list of strings containing stopwords and remove empty elements

arg : 
    t : a list of strings 
    stop_words : function in the NLTK package to remove stopwords
'''

def remove_stop_words(t,stop_words = stopwords.words('english')):
    stop_words.extend(['from', 'use','would','know','way','need','seem','example',
                      'want','try','make','give','get','like','one','set','anyone','x',
                      'go','file','change','code','look','create','question','question',
                      'something','possible','nt','add','see','page','work',
                      'service','option','could'])
    filtered_sentence = [w for w in t if not w in stop_words]
    filtered_sentence = list(filter(None,filtered_sentence))
    return filtered_sentence

test = df['list_body'].copy()
print(test.head())

test1 = test.apply(lambda row : clean_txt(row))
print(test1.head())

test2 = test1.apply(lambda row : remove_stop_words(row))
print(test2.head())

df['list_body'] = df['list_body'].apply(lambda row : clean_txt(row))
df['list_title'] = df['list_title'].apply(lambda row : clean_txt(row))

df['list_body'] = df['list_body'].apply(lambda row : remove_stop_words(row))
df['list_title'] = df['list_title'].apply(lambda row : remove_stop_words(row))


df[['list_body','list_title']].head(10)

# Lemmatisation

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

test = df['list_body'].copy()
test.head()

print(test[0])

'''
Lemmatize function with appropriate POS tag

arg : 
    - word : str to lemmatize
'''
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



df['list_body'] = df['list_body'].apply(lambda row : get_lemmatize_sentence(row))
df['list_title'] = df['list_title'].apply(lambda row : get_lemmatize_sentence(row))

df[['list_body','list_title']].head()

df.head()

df.info()

df_final = df[['Id','Tags_list','list_title','list_body']]
df_final.head()

# Export du dataset clean

df_final.to_csv('dataset_clean.csv')