"""
Primary module for the CS 4780 final project

This module contains the classifier to determine if tweets are from Trump's
Android phone, or one of his advisors' iPhones

# Alexander Li (afl59), Rohan Patel (rp442)
# 11/14/18

# https://www.tensorflow.org/tutorials/keras/basic_text_classification
"""

import numpy as np
import tensorflow as tf
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

tweets = []
length = []
labels = []
time = []
xTr = []


data = csv.DictReader(open('train.csv'))

def hashfeatures(tweet):
    v = np.zeros(1000)
    #for i in range(len(baby)):
    #    index = i*26
    #    offset = ord(baby[i]) - ord('a')
    #    letterindex = 670+index+offset
    #    v[letterindex] = 1

    #if(baby[-1] == 'a'):
        #v[hash('!') % B] = 1
        #v[26] = 1
    #bigrams
    #for i in range(len(baby)-1):
    #    first = (ord(baby[i])-ord('a'))*26+26
    #    second = (ord(baby[i+1])-ord('a'))
    #    bigram = first+second
    #    v[bigram] = 1
    tweetlist = tweet.split()
    for s in tweetlist:
        index = hash(s)%1000
        orig = v[index]
        v[index] = orig + 1
    return v

for d in data:
    tweet = d['text']
    tweets.append(tweet)
    size = len(d['text'])
    length.append(size)
    labels.append(int(d['label']))
    timedate = d['created']
    space = timedate.find(' ')
    justtime = timedate[space+1:]
    hour = int(justtime[:justtime.find(':')])
    time.append(hour)
    tweethash = hashfeatures(tweet)
    #print(tweethash)
    #print(size)
    data=np.append(tweethash,size)
    data = np.append(data,hour)
    xTr.append(data)

xTr = np.array(xTr)
yTr = np.array(labels)

n = len(xTr)

trainxTr = xTr[:int(n*.8)]
validxTr=xTr[int(n*.8):]

trainyTr = yTr[:int(n*.8)]
validyTr=yTr[int(n*.8):]


forest = RandomForestClassifier(n_estimators=500, max_depth=30,random_state=0)
forest.fit(trainxTr, trainyTr)

pred = forest.predict(validxTr)
print('forest')
print(sum(validyTr!=pred)/len(validyTr))


support = svm.SVC(kernel='rbf')
support.fit(trainxTr,trainyTr)
pred = forest.predict(trainxTr)
#<GRADED>
print(sum(trainyTr!=pred)/len(trainyTr))

pred = support.predict(validxTr)
print('svm')
pred = support.predict(validxTr)
print(sum(validyTr!=pred)/len(validyTr))
pred = support.predict(trainxTr)
print(sum(trainyTr!=pred)/len(trainyTr))



def name2features2(filename, B=128, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B)
    return X

#</GRADED>
