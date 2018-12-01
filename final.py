"""
Primary module for the CS 4780 final project

This module contains the classifier to determine if tweets are from Trump's
Android phone, or one of his advisors' iPhones

# Alexander Li (afl59), Rohan Patel (rp442)
# 11/14/18

# https://www.tensorflow.org/tutorials/keras/basic_text_classification
"""

import numpy as np
#import tensorflow as tf
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


tweets = []
length = []
time = []


data = csv.DictReader(open('train.csv'))
test = csv.DictReader(open('test.csv'))

def prepareSubmit(pred):
    with open('submit.csv', mode='w') as submit:
        writer = csv.writer(submit, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'Label'])
        for i in range(len(pred)):
            id = i
            label = pred[i]
            writer.writerow([str(i), str(label)])

def hashfeatures(tweet):
    v = np.zeros(1000)
    tweetlist = tweet.split()
    for s in tweetlist:
        index = hash(s)%1
        #orig = v[index]
        v[index] = 1000
    return v

def extractLabels(data):
    labels = []
    for d in data:
        print('did')
        label = int(d['label'])
        labels.append(label)
        print(label)
    return labels

def extractFeatures(data,test=False):
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

        retweetLarge = (int(d['retweetCount'])>40000)*1
        favoriteLarge = (int(d['favoriteCount'])>10000)*1

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

xTr,yTr = extractFeatures(data)
xTe = extractFeatures(test, True)

xTr = np.array(xTr)
yTr = np.array(yTr)

n = len(xTr)

trainxTr = xTr[:int(n*.8)]
validxTr=xTr[int(n*.8):]

trainyTr = yTr[:int(n*.8)]
validyTr=yTr[int(n*.8):]

forest = RandomForestClassifier(n_estimators=1000, max_depth=None,random_state=0, oob_score=True)
forest.fit(trainxTr, trainyTr)

pred = forest.predict(validxTr)
print('forest')
print('validation score ' + str(1-sum(validyTr!=pred)/len(validyTr)))

pred = forest.predict(trainxTr)
#<GRADED>
print('train score ' + str(1-sum(trainyTr!=pred)/len(trainyTr)))
print('oob score' + str(forest.oob_score_))


forest2 = RandomForestClassifier(n_estimators=1000, max_depth=None,random_state=0, oob_score=True)
forest2.fit(xTr, yTr)

pred = forest2.predict(xTe)

prepareSubmit(pred)

print('final')
print('test oob score' + str(forest.oob_score_))



# support = svm.SVC(kernel='rbf')
# support.fit(trainxTr,trainyTr)
#
#
#
# pred = support.predict(validxTr)
# print('svm')
# pred = support.predict(validxTr)
# print(sum(validyTr!=pred)/len(validyTr))
# pred = support.predict(trainxTr)
# print(sum(trainyTr!=pred)/len(trainyTr))


# def name2features2(filename, B=128, LoadFile=True):
#     """
#     Output:
#     X : n feature vectors of dimension B, (nxB)
#     """
#     # read in baby names
#     if LoadFile:
#         with open(filename, 'r') as f:
#             babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
#     else:
#         babynames = filename.split('\n')
#     n = len(babynames)
#     X = np.zeros((n, B))
#     for i in range(n):
#         X[i,:] = hashfeatures(babynames[i], B)
#     return X

#</GRADED>
