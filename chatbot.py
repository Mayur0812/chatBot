# -*- coding: utf-8 -*-
"""
@author: Mayur.Dalal

"""

import nltk
import numpy as np
import random
import string
import tensorflow as tf
import pandas as pd
import seaborn as sns
import tensorflow_hub as hub
import tensorflow as tf
import warnings
import time 

from scipy import spatial #for KDTree implementation

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING) #to supress tensorflow warnings and INFO

"""### Importing Data :"""

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_excel("drive/My Drive/Colab Notebooks/Chatbot_Questions_Non_integers.xlsx")

data.head()

dataset = data[['Bot questions ', 'Url Details']]

dataset

"""## Pre-processing data:"""

nltk.download('punkt')
nltk.download('wordnet')

#Converting queries to lowercase: 

dataset['Bot questions ']=dataset['Bot questions '].str.lower()

chat_data = list(dataset['Bot questions '])

chat_data

#Converting corpus to list of sentences and then to words:

sent_tokens = nltk.sent_tokenize(chat_data)
word_tokens = nltk.word_tokenize(chat_data)

"""## Using Google's Universal sentence Encoder for creating embeddings"""

embed = hub.Module('https://tfhub.dev/google/universal-sentence-encoder-large/3')

sentences = tf.placeholder(dtype =tf.string, shape=[None])
embedding_fun= embed(chat_data)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# 
# with tf.Session() as session:
#   session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#   message_embeddings = session.run(embedding_fun)
#  
#   for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#     print("Message: {}".format(chat_data[i]))
#     print("Embedding size: {}".format(len(message_embedding)))
#     message_embedding_snippet = ", ".join(
#         (str(x) for x in message_embedding[:3]))
#     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

## Making tree(sorted) of recommendations:
## Using K - Dimensional Tree data structure: 

tree = spatial.KDTree(message_embeddings)

session = tf.Session()

intent_num = None
def fetch_reply(in_sen):
  in_sen= [' '.join(in_sen)]
  message_dec = embed(in_sen)
  
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    dec_msg = session.run(message_dec)
  ans = tree.query(dec_msg[0])
  intent_num = ans[1]
  #print("Intent number is: ", intent_num)
  return(dataset['Url Details'][ans[1]], intent_num)

def embed_GUSE(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
  
embed_fn = embed_GUSE('module_USE')

import re
order_num = None

def order_number_input():
  order_num= input("Enter your order number :")
  if(order_num.isdigit()):
    return order_num
  else:
    print("Invalid order number...!")
    order_number_input()

#input_sen = input("enter your query here....!\n")
def order_number(input_sen):
  temp = re.findall(r'\d+', input_sen)
  if(len(temp)>=1):
    
    if(len(temp)==1):
      #print("Your order number is : ",str(order_in[0]))
      return str(temp[0])
    else:
      print("Enter one order number:..\nYou entered: ",temp)
      order_num1=order_number_input()
      print("Your order number here is :", order_num)
      return order_num1
  else:
    order_num =order_number_input()
    #print("Your order number is :", order_num)
    return order_num

print("Speak a sentence to be recorded and translated: ")
r = sr.Recognizer()
with sr.Microphone() as source:
    audio = r.listen(source)
	
input_voice = r.recognize_google(audio)
#print("type is  ",type(input_voice))
print(input_voice)

trans = Translator()
text = trans.translate(input_voice , dest = 'en')
print("Detected source language is:", text.src)
print("Translated speech in english is: ",text.text)

flag = True
intent = None
sent = None
order_num = None

def take_input():
  flag = True
  while(flag == True):
    print("\n=========================\n\n")
    sent = input("Enter your query: ")
    if(sent!='bye'):
      start = time.time()
      ans,intent = fetch_reply(sent.split())
      end = time.time()
      print("Time taken is :", end - start)
      print("Intent is: ", intent)
      if(intent == 1 or intent ==2):
        global order_num
        order_num = order_number(sent)
        print("Your order number is: ", order_num)
        new_url = ans + "+for=" + order_num
        print("URL will be redirected to: ", new_url)
        #print("BOT : ",ans)
      elif(intent == 13):
        if(order_num==None):
          order_num = order_number(sent)
          print("Your order number is: ",order_num)
          new_url = ans + "+for=" + order_num
          print("checking ETA for order :", new_url)
        else:
          print("Do you want to see ETA for the order :", order_num)
          print("Enter Yes or No")
          choice_input = str(input())
          if(choice_input == 'y' or choice_input == "Y" or choice_input == "Yes" or choice_input == "yes"):
            new_url = ans + "+for=" + order_num
            print("checking ETA for order :", new_url)
          else:
            order_temp = input("Enter order number  :")
            new_url = ans + "+for=" + order_temp
            print("checking ETA for order :", new_url)
      elif(intent==3):
        print("Checking current LME price: \n")
        print("URL will be redirected to: ",ans)
      elif(intent==4):
        print("Checking historical LME price: ")
        print("URL will be redirected to: ",ans)
      elif(intent==5):
        print("Showing active standing instructions: ")
        print("URL will be redirected to: ",ans)
      elif(intent==6):
        print("Showing expired standing instructions: ")
        print("URL will be redirected to: ",ans)
      elif(intent==7):
        print("Showing cancelled standing instructions: ")
        print("URL will be redirected to: ",ans)
      elif(intent==9):
        print("Price settings you performed last week are: ")
        print("URL will be redirected to: ",ans)
      elif(intent==10):
        print("Showing number of orders you declared: ")
        print("URL will be redirected to: ",ans)
             
          
          
    else:
      flag = False
      print("BOT : thanks for visiting")


take_input()

dataset

"""## ChatBot using ChatterBot"""

! pip install chatterbot
! pip install --upgrade chatterbot_corpus

! pip install --upgrade ListTrainer

import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

conversation =(['what is your name','My name is EGA_Bot'])

trainer = ListTrainer(chatbot)

trainer.train(conversation)

#chatbot.get_response("your name..?")
chatbot.get_response("who is EGA_Bot")



def match_intent():

temp = input()
temp.isdigit()

in_sen= input("Enter your sentence: ").split()
in_sen= [' '.join(in_sen)]
print("input sentence is: ", in_sen)

with tf.Session() as session:
  session.run([tf.global_variables_initializer(), tf.tables_initializer()])
  message_dec = session.run(embed(in_sen))

#print("embedding is: ", message_dec)

tree.query(message_dec[0])
recommendations = []
query_out=tree.query(message_dec[0])
print("Question : ", dataset['Bot questions '][query_out[1]])
print("Answer : ", dataset['Response'][query_out[1]])

Greeting_inputs = ["hello","hi", "greetings" , "sup","what's up", "hey","namaste"]
Greetings_responses = ["Hey there...!","Hello there", "glad to see you here, welcome :)", "Hello","hii"]

def greeting(sentence):
  for word in sentence.split():
    if word.lower() in Greeting_inputs:
      return random.choice(Greetings_responses)

"""## Implementing Google's USE by reducing overhead:"""

!mkdir module_USE
!curl -L "https://tfhub.dev/google/universal-sentence-encoder-large/3?tf-hub-format=compressed" | tar -zxvC module_elmo2

### Defining a function :

def embed_GUSE(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})

embed_fn = embed_GUSE('module_elmo2')

in_sen = input("Enter your query").split()
res = embed_fn(in_sen)

def get_reply(in_sen):
  in_sen= [' '.join(in_sen)]
  message_dec = embed_fn(in_sen)
  query_out=tree.query(res)
  return(list(dataset['Response'][query_out[1]]))

flag = True
while(flag == True):
  sent = input("Enter your query: ")
  if(sent!='bye'):
    ans = get_reply(sent.split())
    print("BOT : ",ans)
  else:
    flag = False
    print("BOT : thanks for visiting")

query_out=tree.query(res)

list(dataset['Bot questions '][query_out[1]])

query_out=tree.query(res)
print("Question : ", list(dataset['Bot questions '][query_out[1]]))
print("Answer : ", list(dataset['Response'][query_out[1]]))