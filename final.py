"""
Primary module for the CS 4780 final project

This module contains the classifier to determine if tweets are from Trump's
Android phone, or one of his advisors' iPhones

Alexander Li (afl59), Rohan Patel (rp442)
11/14/18

"""

import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import re

BAGSIZE = 1000

data = csv.DictReader(open('train.csv')) #training data
test = csv.DictReader(open('test.csv')) #test submission data

def prepareSubmit(pred):
    """
    Writes the test predictions to a CSV for submission
    Parameter pred: list of classifications for the test data.
    """
    with open('submit.csv', mode='w') as submit:
        writer = csv.writer(submit, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'Label'])
        for i in range(len(pred)):
            id = i
            label = pred[i]
            writer.writerow([str(i), str(label)])

def hashfeatures(tweet):
    """
    Creates a bag of words representation of the Tweets. Cleans up the tweet
    to all lower case words only, and then hashes them to a feature vector.
    Parameter tweet: string of the tweet to hash
    """
    v = np.zeros(BAGSIZE)
    tweet = re.sub(r'([^\s\w]|_)+', '', tweet)
    tweet = tweet.lower()
    tweetlist = tweet.split()
    for s in tweetlist:
        index = hash(s)%BAGSIZE
        #orig = v[index]
        v[index] = 1
    return v

def extractLabels(data):
    """
    Creates the labels used for training the models
    Parameter data: dictionary of data from CSV
    """
    labels = []
    for d in data:
        print('did')
        label = int(d['label'])
        labels.append(label)
        print(label)
    return labels

def extractFeatures(data,test=False):
    """
    Extracts features for the feature vector used to train the models.
    Creates a bag of words, and adds features for special characters and
    words of interest.
    Parameter data: dictionary of data from CSV
    Parameter test: boolean to distinguish if this is for training or testing
    """
    xTr = []
    yTr = []
    for d in data:
        if not test:
            label = d['label']
            yTr.append(int(d['label']))
        tweet = d['text']
        containsI = 'I ' in tweet
        containsI = containsI*1

        containsHash = '#' in tweet
        containsHash = containsHash*1

        containsLink = 'https' in tweet
        containsLink = containsLink*1

        containsAt = '@' in tweet
        containsAt = containsAt*1

        containsMedia = 'media' in tweet or 'Media' in tweet
        containsMedia = containsMedia*1

        containsHillary = 'Hillary' in tweet
        containsHillary = containsHillary*1

        containsTrump = 'me' in tweet or 'Me' in tweet
        containsTrump = containsTrump*1

        containsDots = '...' in tweet
        containsDots = containsDots*1

        containsThank = 'Thank' in tweet or 'thank' in tweet
        containsThank = containsThank*1

        size = len(d['text'])

        timedate = d['created']
        space = timedate.find(' ')
        justtime = timedate[space+1:]
        hour = int(justtime[:justtime.find(':')])
        evening = (hour > 18)*1
        morning = (hour < 7)*1
        tweethash = hashfeatures(tweet)

        retweetLarge = (int(d['retweetCount'])>10000)*1
        favoriteLarge = (int(d['favoriteCount'])>30000)*1

        data=np.append(tweethash,size)
        data = np.append(data,evening)
        data = np.append(data,morning)
        data = np.append(data,retweetLarge)
        data = np.append(data,favoriteLarge)
        data = np.append(data,containsI)
        data = np.append(data,containsHash)
        data = np.append(data,containsLink)
        data = np.append(data,containsAt)
        data = np.append(data,containsMedia)
        data = np.append(data,containsHillary)
        data = np.append(data,containsTrump)
        data = np.append(data,containsDots)
        data = np.append(data,containsThank)
        xTr.append(data)
    if test:
        return xTr
    else:
        return xTr,yTr

def train(xTr, yTr):
    """
    Trains a Random Forest Model on our data
    Parameter xTr: training data
    Parameter yTr: training labels
    """
    forest = RandomForestClassifier(n_estimators=1000, max_depth=None,random_state=0, oob_score=True)
    forest.fit(trainxTr, trainyTr)
    return forest

# Generate training data and labels
xTr,yTr = extractFeatures(data)
xTe = extractFeatures(test, True)



# Split data into validation and testing
n = len(xTr)
trainxTr = xTr[:int(n*.8)]
trainyTr = yTr[:int(n*.8)]
validxTr = xTr[int(n*.8):]
validyTr = yTr[int(n*.8):]

# Create model and test the model
forest = train(trainxTr, trainyTr)
predValid = forest.predict(validxTr)
predTrain = forest.predict(trainxTr)

print('forest')
print('validation score ' + str(1-sum(validyTr!=predValid)/len(validyTr)))

print('train score ' + str(1-sum(trainyTr!=predTrain)/len(trainyTr)))
print('oob score' + str(forest.oob_score_))


# Final model for submission
forest2 = train(xTr,yTr)
pred = forest2.predict(xTe)

prepareSubmit(pred)

print('final')
print('test oob score' + str(forest.oob_score_))
