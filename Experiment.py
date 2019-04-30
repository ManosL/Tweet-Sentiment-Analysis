# Import all libraries needed for the tutorial
import os

import numpy as np

import nltk
from nltk import pos_tag

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

# Vectorization Libraries

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

import gensim

from pickle import load,dump

######################

# Classification Libarries

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score

##########################
# Enable inline plotting
#%matplotlib inline

# removed @ and #
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~'

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

def unify_negations(words):
    new_words = []

    for word in words:
        if word == "n't":
            new_words_len = len(new_words)
            new_words[new_words_len - 1] = new_words[new_words_len - 1] + "n't"
        else:
            new_words.append(word)

    return new_words

def readLexica():
    affin_words = set()
    valence_tweet_words = set()
    generic_words = set()
    nrc_val_words = set()
    nrctag_val_words = set()

    affin_weights = {}
    valence_tweet_weights = {}
    generic_weights = {}
    nrc_val_weights = {}
    nrctag_val_weights = {}

    filePaths = ['../lexica/affin/affin.txt','../lexica/emotweet/valence_tweet.txt',
    '../lexica/generic/generic.txt','../lexica/nrc/val.txt','../lexica/nrctag/val.txt']

    sets = [affin_words,valence_tweet_words,generic_words,
            nrc_val_words,nrctag_val_words]
    
    weights = [affin_weights,valence_tweet_weights,generic_weights,
            nrc_val_weights,nrctag_val_weights]

    for fileName,set_,weight_dict in zip(filePaths,sets,weights):
        file = open(fileName,'r')

        fileLines = file.readlines()

        for line in fileLines:
            linelist = line.split(' ')
            linelist2 = []

            for token in linelist:
                linelist2 += token.split('\t')
            
            linelist = linelist2

            word = ' '.join(linelist[0:len(linelist)-1])

            if word not in set_:
                set_.add(word)
                weight_dict[word] = float(linelist[len(linelist) - 1])

            file.close()
    
    return tuple(weights)


affin_dict,valence_tweet_dict,generic_dict,nrc_val_dict,nrctag_val_dict = readLexica()

train_set_path = '../twitter_data/train2017.tsv'
test_set_path  = '../twitter_data/test2017.tsv'
test_set_gold_path  = '../twitter_data/SemEval2017_task4_subtaskA_test_english_gold.txt'

filesToRead = [train_set_path,test_set_path,test_set_gold_path]
dataframes  = []

for filePath in filesToRead:
    file = open(filePath,"r")

    line = file.readline()
    lines = []

    while len(line) != 0:
        line = line.split("\t")

        line[len(line) - 1] = line[len(line) - 1].replace('\n','')

        if len(line) > 0:
            lines.append(line)

        line = file.readline()

    if filePath != test_set_gold_path:
        curr_df = pd.DataFrame(data = lines,columns = ['ID1','ID2','Tag','Tweet'])
        curr_df = curr_df[['Tag','Tweet']]
        dataframes.append(curr_df)
    else:
        curr_df = pd.DataFrame(data = lines,columns = ['ID1','Tag'])   
        curr_df = curr_df[['Tag']]        # I suppose that original tags are written with the
                                          # same sequence as tweets at file
        dataframes.append(curr_df)

    file.close()

training_dataframe = dataframes[0]
test_dataframe = dataframes[1]
print(test_dataframe)
test_solutions = dataframes[2]

# Getting only the 2 columns that we need

training_dataframe = training_dataframe[['Tag','Tweet']]

print(training_dataframe)

"""
print("Positive: %d\nNegative: %d\nNeutral: %d\n" % 
            (len(training_dataframe[training_dataframe['Tag'] == 'positive']),
            len(training_dataframe[training_dataframe['Tag'] == 'negative']),
            len(training_dataframe[training_dataframe['Tag'] == 'neutral'])))
"""

#COUNT example
tags = training_dataframe['Tag'].value_counts()
neutral_num = tags['neutral']
positive_num = tags['positive']
negative_num = tags['negative']

print("Positive Tweets Percentage %.1f" % ((float(positive_num) / len(training_dataframe))*100))
print("Negative Tweets Percentage %.1f" % ((float(negative_num) / len(training_dataframe))*100))
print("Neutral Tweets Percentage %.1f" % ((float(neutral_num) / len(training_dataframe))*100))

# Printing the bar plot for tweets

tags.plot.bar(title = 'Tweets sentimel tendency')
plt.xticks(rotation = 0)

matplotlib.pyplot.show()

# EXAMPLE CLEANUP(CONVERTS TWEETS TO LOWER CASE)
training_dataframe['Tweet'] = training_dataframe.Tweet.apply(lambda t: t.lower())

##############################################
# CLEANUP PHASE 

cleaned_tweets = []

for tweet in training_dataframe['Tweet']:
    tweet = tweet.split(' ')

    tweet = filter(remove_unnecessary_words,tweet) # Removing hashtags,tags and links
    
    tweet = ' '.join(tweet)

    cleaned_tweets.append(tweet)

training_dataframe['Tweet'] = cleaned_tweets
cleaned_tweets = []

all_adjs_and_verbs = []
all_adjs_and_verbs_pos = []

all_adjs_and_verbs_neutral = []
neutral_adjs_and_verbs = set()

all_adjs_and_verbs_neg = []

for tweet_tag,tweet in zip(training_dataframe['Tag'],training_dataframe['Tweet']):
    original_tweet = tweet
    for letter in tweet:
        if letter in my_punctuation:
            for char in my_punctuation:
                if letter == char:
                    tweet = tweet.replace(char,' ')

    splitted_tweet = nltk.word_tokenize(tweet) 
    splitted_tweet = unify_negations(splitted_tweet)  #I do that because splits words like can't

    cleaned_tweet = []


    for word in splitted_tweet:             #Removing punctuation
        cleaned_tweet.append(word.strip(my_punctuation))

            
    tweet = [word for word in cleaned_tweet if word not in nltk.corpus.stopwords.words('english')
                                                        and len(word) > 0]

    pos_tags = pos_tag(tweet)

    for tag in pos_tags: 
        if (tag[1][0] == 'J' or tag[1][0] == 'V') and len(tag[0]) > 1:
            all_adjs_and_verbs.append(tag[0])

            if tweet_tag == 'positive':
                if tag[0] not in neutral_adjs_and_verbs:
                    all_adjs_and_verbs_pos.append(tag[0])
            elif tweet_tag == 'negative':
                if tag[0] not in neutral_adjs_and_verbs:
                    all_adjs_and_verbs_neg.append(tag[0])
            else:
                all_adjs_and_verbs_neutral.append(tag[0])
                neutral_adjs_and_verbs.add(tag[0])
        
    tweet = ' '.join(tweet)
    
    cleaned_tweets.append(tweet)

all_adjs_and_verbs_text = ' '.join(all_adjs_and_verbs)

cloud = wordcloud.WordCloud().generate(all_adjs_and_verbs_text)

plt.title("All Words")
plt.imshow(cloud,interpolation = 'bilinear')
plt.axis("off")
plt.show()

all_adjs_and_verbs_text = ' '.join(all_adjs_and_verbs_pos)

cloud = wordcloud.WordCloud().generate(all_adjs_and_verbs_text)

plt.title("Positive Words")
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
plt.show()

all_adjs_and_verbs_text = ' '.join(all_adjs_and_verbs_neg)

cloud = wordcloud.WordCloud().generate(all_adjs_and_verbs_text)

plt.title("Negative Words")
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
plt.show()

all_adjs_and_verbs_text = ' '.join(all_adjs_and_verbs_neutral)

cloud = wordcloud.WordCloud().generate(all_adjs_and_verbs_text)

plt.title("Neutral Words")
plt.imshow(cloud,interpolation='bilinear')
plt.axis("off")
plt.show()

training_dataframe['Tweet'] = cleaned_tweets
print(training_dataframe)
###############################################

# Getting the 20 most common words at negative tweets

negative_tweets_words = []

#sent_tokenize for sentence tokenize

eng_stopwords = nltk.corpus.stopwords.words('english')

for sentence in training_dataframe['Tweet']:#[training_dataframe.Tag == 'negative']:
    tokenized = sentence.split(' ')
    negative_tweets_words += tokenized

count = nltk.Counter(negative_tweets_words)

most_common = count.most_common(20)

print(most_common)

plt.bar([x[0] for x in most_common],[x[1] for x in most_common],data = most_common)
plt.show()

####################################

######### VECTORIZATION ############

# Bag Of Words

bow_vectorizer = CountVectorizer(max_features = 1000,stop_words = 'english')

bow_xtrain = bow_vectorizer.fit_transform(training_dataframe['Tweet'])

print(bow_xtrain.shape)

# TF-IDF

tfidf_vectorizer = TfidfVectorizer(max_features = 1000,stop_words = 'english')

tfidf = tfidf_vectorizer.fit_transform(training_dataframe['Tweet'])

print(tfidf.shape)

# Word2Vec

if not os.path.isfile('./pickle_files/model_w2v.pkl'):
    tokenized_tweet = training_dataframe['Tweet'].apply(lambda x: x.split()) # tokenizing 
    model_w2v = gensim.models.Word2Vec(
                tokenized_tweet,
                size=200, # desired no. of features/independent variables
                window=5, # context window size
                min_count=2,
                sg = 1, # 1 for skip-gram model
                hs = 0,
                negative = 10, # for negative sampling
                workers= 2, # no.of cores
                seed = 34) 

    model_w2v.train(tokenized_tweet, total_examples= len(training_dataframe['Tweet']), epochs=20)

    dump(model_w2v,open("pickle_files/w2v_model.pkl","wb"))
else:
    model_w2v = load(open("pickle_files/w2v_model.pkl","rb"))

# WRITE VECTORS TO PICKLE FILE

print(model_w2v.wv.most_similar(positive="trump"))

def W2V_TweetVectorize(tweets,w2v_model):
    vectors = []
    vector = np.zeros(w2v_model.wv.vector_size)

    for tweet in tweets:
        vector_words_num = 0
        splitted_tweet = nltk.word_tokenize(tweet)

        for word in splitted_tweet:
            if word in w2v_model.wv.vocab:
                vector += w2v_model[word]
                vector_words_num += 1

        if vector_words_num == 0:
            print(tweet)
            vectors.append(vector)
        else:
            vectors.append(vector / vector_words_num)

    return vectors 

if not os.path.isfile('./pickle_files/w2v_train_vectors.pkl'):
    w2v_train_vectors = W2V_TweetVectorize(training_dataframe['Tweet'],model_w2v)

    dump(w2v_train_vectors,open("pickle_files/w2v_train_vectors.pkl","wb"))
else:
    w2v_train_vectors = load(open("pickle_files/w2v_train_vectors.pkl","rb"))

def tsne_plot(model,words_to_plot):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    for word in list(model.wv.vocab)[0:words_to_plot + 1]:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []

    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords='offset points',
                ha='right',
                va='bottom')
    
    plt.show()

    return

words_num_to_plot = 1000
tsne_plot(model_w2v,words_num_to_plot)

# TO ADD (PROBABLY) EXTRA FEATURES
####################################

########## CLASSIFICATION ##########

def SVM_Classifier(train_vectors,train_labels,test_vectors,test_labels):
    xtrain, xvalid, ytrain, yvalid = train_test_split(train_vectors[0:1000], 
                                        train_labels[0:1000],
                                        random_state=42, test_size=0.2)

    svc = svm.SVC(kernel='linear', C=1, probability=True)
    svc = svc.fit(xtrain, ytrain) # xtrain:bag of words features for train data, ytrain: train data labels
    prediction = svc.predict(xvalid) #predict on the validation set

    F1_Score = f1_score(yvalid,prediction,average = 'micro') #evaluate on the validation set

    correct_num = 0
    for tag,yval in zip(tags,yvalid):
        if yval == tag:
            correct_num += 1

    percentage = (float(correct_num) / float(len(yvalid)))

    return (percentage,F1_Score)

def KNN_Classifier(train_vectors,train_labels,test_vectors,test_labels):
    xtrain, xvalid, ytrain, yvalid = train_test_split(train_vectors[0:1000], 
                                        train_labels[0:1000],
                                        random_state=42, test_size=0.2)   

    knn_classifier = KNeighborsClassifier(n_neighbors = 5)

    knn_classifier = knn_classifier.fit(xtrain,ytrain)
    prediction = knn_classifier.predict(xvalid)

    F1_Score = f1_score(yvalid,prediction,average = 'micro')

    correct_num = 0
    for tag,yval in zip(prediction,yvalid):
        if tag == yval:
            correct_num += 1
    
    percentage = (float(correct_num)/float(len(yvalid)))

    return (percentage,F1_Score)

print(SVM_Classifier(bow_xtrain,training_dataframe['Tag'],None,None))
print(SVM_Classifier(tfidf,training_dataframe['Tag'],None,None))
print(SVM_Classifier(w2v_train_vectors,training_dataframe['Tag'],None,None))

print(KNN_Classifier(bow_xtrain,training_dataframe['Tag'],None,None))
print(KNN_Classifier(tfidf,training_dataframe['Tag'],None,None))
print(KNN_Classifier(w2v_train_vectors,training_dataframe['Tag'],None,None))

####################################