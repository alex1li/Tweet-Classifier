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

tweets = []
length = []
labels = []
time = []

for d in csv.DictReader(open('train.csv')):
    tweets.append(d['text'])
    length.append(len(d['text']))
    labels.append(int(d['label']))
    time.append(d['created'])
print(created)
