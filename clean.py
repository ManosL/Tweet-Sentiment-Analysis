# Import all libraries needed for the tutorial
import os

import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import PorterStemmer

# General syntax to import specific functions in a library: 
##from (library) import (specific library function)
from pandas import DataFrame, read_csv

# General syntax to import a library but no functions: 
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

import wordcloud

import re

def remove_unnecessary_words(str):
    #FIND A WAY TO REMOVE EMOJIS
    if len(str) > 0 and str[0] != '@' and str[0] != '#':
        if len(str) > 4 and str[0:4] != 'http':
            return True
        elif len(str) <= 4:
            return True
        else:
            return False
    else:
        return False


def stop_words(stemmer,tweet):
    
    tweet = [ stemmer.stem(word) for word in tweet if word not in stopwords.words('english') and len(word) > 1]
    return tweet

my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'

location = '../twitter_data/train2017.tsv'

train = pd.read_csv(location,sep="\t",names=['ID1','ID2','Tag','Tweet'])

print(train.Tweet)

tags = train['Tag'].value_counts()
neutral_num = tags['neutral']
positive_num = tags['positive']
negative_num = tags['negative']

print("Positive Tweets Percentage %.1f" % ((float(positive_num) / len(train))*100))
print("Negative Tweets Percentage %.1f" % ((float(negative_num) / len(train))*100))
print("Neutral Tweets Percentage %.1f" % ((float(neutral_num) / len(train))*100))

tags.plot.bar(title = 'Tweets sentimel tendency')
plt.xticks(rotation = 0)

matplotlib.pyplot.show()

train['Tweet'] = train.Tweet.apply(lambda t: t.lower())

#re_punctuation = r"[{}]".format(my_punctuation)
train['Tweet'] = train.Tweet.apply(lambda t: re.sub(r'[^a-zA-Z#@ ]',"",t))

#train['Tweet'] = train.Tweet.apply(lambda t: filter(remove_unnecessary_words,t))

train['Tweet'] = train.Tweet.apply(lambda t: re.sub("@[a-zA-Z]+","",t))
train['Tweet'] = train.Tweet.apply(lambda t: re.sub("#[a-zA-Z]+","",t))
train['Tweet'] = train.Tweet.apply(lambda t: re.sub("http[a-zA-Z]+","",t))



train['Tweet'] = train.Tweet.apply(lambda t: nltk.word_tokenize(t) )

stemmer = PorterStemmer()
train['Tweet'] = train.Tweet.apply(lambda t:  ' '.join( stop_words(stemmer,t) ))

print(train.Tweet)

train.to_csv('test.csv',index=False,header=False)
