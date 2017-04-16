# author Panagiotis Theodorakakos

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import codecs
import xml.etree.ElementTree as ET
import absa2016_lexicons
import absa2016_embeddings

# UNCOMMENT TO USE EMBEDDINGS MODEL
# import gensim
# from gensim.models import Word2Vec
# m = gensim.models.Word2Vec.load('model.bin') #load the model


# ---------- use the helpingEmbeddings.txt instead of the embeddings model ----------
def load_helping_embeddings():
    csv = np.loadtxt('helpingEmbeddings.csv',
                     delimiter=',',
                     dtype=np.dtype('S32, 200f8'))
    return csv

helping_embeddings = load_helping_embeddings()

train_path = 'restaurants/ABSA16_Restaurants_Train_SB1_v2.xml'
test_path = 'restaurants/EN_REST_SB1_TEST.xml.gold'
domain = 'rest'

print('-------- Features Model--------')
fea2 = absa2016_lexicons.features(train_path, test_path, domain)
train_vector, train_tags = absa2016_lexicons.features.train(fea2, domain)
test_vector = absa2016_lexicons.features.test(fea2, domain)
predictions1 = absa2016_lexicons.features.results(fea2, train_vector, train_tags, test_vector, domain)
print 'End Features Model'

print('-------- Embeddings Model--------')
fea2 = absa2016_embeddings.features(train_path, test_path, helping_embeddings)
train_vector, train_tags = absa2016_embeddings.features.train(fea2)
test_vector = absa2016_embeddings.features.test(fea2)
predictions2 = absa2016_embeddings.features.results(fea2, train_vector, train_tags, test_vector)
print 'End Embeddings Model'

# both methods "vote"
l = len(predictions1)
predictionLabels1 = []
predictionLabels2 = []
predictionLabels = []

w1, w2 = 0.5, 0.5
labels = [
    'negative',
    'neutral',
    'positive',
]

for i in range(l):
    prediction1 = labels[np.argmax(predictions1[i])]
    predictionLabels1.append(prediction1)

    prediction2 = labels[np.argmax(predictions2[i])]
    predictionLabels2.append(prediction2)

    prediction = [
        float(predictions1[i][0] * w1 + predictions2[i][0] * w2) / 2,
        float(predictions1[i][1] * w1 + predictions2[i][1] * w2) / 2,
        float(predictions1[i][2] * w1 + predictions2[i][2] * w2) / 2
    ]

    prediction = labels[np.argmax(prediction)]
    predictionLabels.append(prediction)

reviews = ET.parse(test_path).getroot().findall('Review')


def calc_accuracy(predictions):
    counter = 0
    trues = 0

    for review in reviews:
        sentences = review[0]
        for sentence in sentences:
            if len(sentence) > 1:
                opinions = sentence[1]
                if len(opinions) > 0:  # check if there are aspects
                    for i in range(len(opinions)):
                        actual_polarity = opinions[i].attrib['polarity']
                        predicted_polarity = predictions[counter]
                        counter += 1

                        if predicted_polarity == actual_polarity:
                            trues += 1

    accuracy = float(trues) / counter
    return accuracy

print('Accuracy1: {}'.format(calc_accuracy(predictionLabels1)))
print('Accuracy2: {}'.format(calc_accuracy(predictionLabels2)))
print('Accuracy: {}'.format(calc_accuracy(predictionLabels)))
