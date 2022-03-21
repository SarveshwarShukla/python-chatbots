# Simple chatbot application using python

import numpy as np # used for numerical computations
import nltk # library for natural language processing
import string # library to handle strings efficiently
import random # random module
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open("chatbot.txt", "r", errors="ignore")
raw_doc = f.read()
raw_doc = raw_doc.lower() # converts the text to lower case (preprocessing)
nltk.download("punkt") # used to tokenize the given text
nltk.download("wordnet") # wordnet is a large word database of english noun
sent_tokens = nltk.sent_tokenize(raw_doc) # converts doc to list of sentences
word_tokens = nltk.word_tokenize(raw_doc) # converts doc to list of words

# Text Processing
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Defining a greeting functions
Greet_inputs = ("hello", "hi", "greetings")
greet_responses = ("hi" "hey", "hi there", "hello")
def greet(sentence):
    for word in sentence.split():
        if word.lower() in Greet_inputs:
            return random.choice(greet_responses)

# Response generation
def response(user_response):
    robo1_response = ""
    TfidVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo1_response = robo1_response + "I am sorry! I dont understand youo"
        return robo1_response
    else:
        robo1_response = robo1_response + sent_tokens[idx]
        return robo1_response
