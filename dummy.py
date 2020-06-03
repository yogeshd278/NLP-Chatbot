import nltk
import io
import numpy as np
import random
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 


## import file
f = open('python.txt', 'r', errors = 'ignore')
raw = f.read()
raw = raw.lower() # convert to lowercase
nltk.download('punkt') # first time use only
nltk.download('wordnet') # first time use only
sent_tokens = nltk.sent_tokenize(raw)  # convert to list of sentences
word_tokens = nltk.word_tokenize(raw)  # convert to list of words

#print('sentence => ', sent_tokens[:2])
#print('word => ', word_tokens[:2])

## preprocessing
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(token) :
    return [lemmer.lemmatize(token) for token in token]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text) :
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

## greeting
GREETING_INPUTS = ('hello','hi','greetings','supeb','hey')
GREETING_RESPONSES = ['hi','hey','how are you','hi there','hello','I am glade!']

def greeting(sentence) :
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


## response
def response(user_response) :
    chatbot_response = ''
    sent_tokens.append(user_response)

    IfidfVec    = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')
    tfidf       = IfidfVec.fit_transform(sent_tokens)
    vals        = cosine_similarity(tfidf[-1], tfidf)
    idx         = vals.argsort()[0][-2]
    flat        = vals.flatten()
    flat.sort()
    req_tdidf = flat[-2]

    if(req_tdidf == 0) :
        chatbot_response += 'I am sorry! I do not understant you.'
        return chatbot_response
    else :
        chatbot_response += sent_tokens[idx]
        return chatbot_response


## chatbot 
flag = True
print('Chatbot: I am chatbot. I will answer your queries about python. If you want to exit, type Bye!')

while(flag == True) :
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye') :
        if(user_response == 'thanks' or user_response == 'thank you') :
            flag = False
            print('Chatbot: You are welcome')
        else :
            if(greeting(user_response) != None) :
                print('Chatbot: ', greeting(user_response))
            else :
                print('Chatbot: ', end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else :
        flag = False
        print('Chatbot: Bye! take care..')






