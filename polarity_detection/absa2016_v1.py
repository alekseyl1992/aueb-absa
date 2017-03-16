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

print '---------------- Laptops ----------------'
print

print('-------- Features Model--------')
fea2 = absa2016_lexicons.features('laptops/ABSA16_Laptops_Train_SB1_v2.xml', 'laptops/EN_LAPT_SB1_TEST_.xml', 'lap')
train_vector, train_tags = absa2016_lexicons.features.train(fea2, 'lap')
test_vector = absa2016_lexicons.features.test(fea2, 'lap')
predictionsLap1 = absa2016_lexicons.features.results(fea2, train_vector, train_tags, test_vector, 'lap')
print 'End Features Model'

print('-------- Embeddings Model--------')
fea2 = absa2016_embeddings.features('laptops/ABSA16_Laptops_Train_SB1_v2.xml', 'laptops/EN_LAPT_SB1_TEST_.xml', helping_embeddings)
train_vector, train_tags = absa2016_embeddings.features.train(fea2)
test_vector = absa2016_embeddings.features.test(fea2)
# store probabilities for each of the three class for each sentence
predictionsLap2 = absa2016_embeddings.features.results(fea2, train_vector, train_tags, test_vector)
print 'End Embeddings Model'

# both methods "vote"
l = len(predictionsLap1)
predictionsLap = []
w1, w2 = 0.5, 0.5
for i in range(l):
    negative = float(predictionsLap1[i][0] * w1 + predictionsLap2[i][0] * w2) / 2  # number of the methods we are using
    neutral = float(predictionsLap1[i][1] * w1 + predictionsLap2[i][1] * w2) / 2
    positive = float(predictionsLap1[i][2] * w1 + predictionsLap2[i][2] * w2) / 2

    if negative > neutral and negative > positive:
        predictionsLap.append('negative')  # check the probabilities
    elif neutral > negative and neutral > positive:
        predictionsLap.append('neutral')
    elif positive > negative and positive > neutral:
        predictionsLap.append('positive')

# creating the xml
r = []  # store the rid the sid's the text the categories and the polarities

reviews = ET.parse('laptops/EN_LAPT_SB1_TEST_GOLD.xml').getroot().findall('Review')

counter = 0
trues = 0

for review in reviews:
    flag = False
    rid = review.attrib['rid']  # get the review id
    sentences = review[0]  # get the sentences
    sid = []
    text = []  # store the text
    cat = []
    for sentence in sentences:
        if len(sentence) > 1:
            opinions = sentence[1]
            if len(opinions) > 0:  # check if there are aspects
                flag = True

                sid.append(sentence.attrib['id'])  # get the sentence id
                text.append(sentence[0].text)  # get the text
                category = []  # store the category
                pr_polarity = []  # store the predicted polarity
                for i in range(len(opinions)):
                    actual_polarity = opinions[i].attrib['polarity']
                    predicted_polarity = predictionsLap[counter]
                    counter += 1

                    if predicted_polarity == actual_polarity:
                        trues += 1

                    category.append(opinions[i].attrib['category'])  # get the category
                    if i == len(opinions) - 1:
                        cat.append(category)
    if flag:
        r.append([rid, sid, text, cat])

accuracy = float(trues) / counter
print('Accuracy: {}'.format(accuracy, trues, counter))


def generate_xml(reviews):
    counter = 0
    with codecs.open('AUEB-ABSA_LAPT_EN_B_SB1_3_1_U.xml', 'w') as o:
        o.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
        o.write('<Reviews>\n')
        for review in reviews:
            o.write('\t<Review rid="%s">\n' % (review[0]))
            o.write('\t\t<sentences>\n')
            for i in range(len(review[1])):
                o.write('\t\t\t<sentence id="%s">\n' % (review[1][i]))
                o.write('\t\t\t\t<text>%s</text>\n' % (review[2][i]))
                o.write('\t\t\t\t<Opinions>\n')
                for j in range(len(review[3][i])):
                    o.write('\t\t\t\t\t<Opinion category="%s" polarity="%s" \t/>\n' % (
                        review[3][i][j], predictionsLap[counter]))
                    counter += 1
                o.write('\t\t\t\t</Opinions>\n')
                o.write('\t\t\t</sentence>\n')
            o.write('\t\t</sentences>\n')
            o.write('\t</Review>\n')
        o.write('</Reviews>')
