#!/usr/bin/env python
# coding: utf-8

# In[370]:

from __future__ import division
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
from collections import Counter
from codecs import open
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import accuracy_score,\
    confusion_matrix,\
    precision_score,\
    recall_score,\
    f1_score, \
    classification_report
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

import numpy as np
import string
import matplotlib.pyplot as plt


# READ DOCS (CODE GIVEN IN ASSIGNMENT)
def read_documents(doc_file):
    docs = []
    labels = []
    with open(doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels


# SET SPLIT FOR TRAINING SET VS EVALUATION SET
# -------------------------- PUT PATH TO DATA SET HERE ----------------
all_docs, all_labels = read_documents("all_sentiment_shuffled.txt")
# ---------------------------------------------------------------------
split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]


# CONVERTS 2D ARRAYS TO STRINGS
def d2array_to_string(arr):
    stringArray = []
    x = range(len(arr))
    for i in x:
        someString = ""
        for word in arr[i]:
            separator = ''
            if i != len(arr):
                separator = ' '
            someString += word + separator
        stringArray.append(someString)
    return stringArray


# REMOVES PUNCTUATION AND THEN REMOVES NEUTRAL WORDS
def text_cleaning(a):
    a = d2array_to_string(a)
    sw = stopwords.words('english')
    sentences = []
    for line in a:
        str = ''
        for char in line:
            if char not in string.punctuation:
                str += char
        sentences.append(str)
    filtered_sentences = []
    for sent in sentences:
        str_arr = []
        for word in sent.split():
            if word.lower() not in sw:
                str_arr.append(word)
        filtered_sentences.append(str_arr)
    return d2array_to_string(filtered_sentences)


# APPLY TEXT CLEANING TO TRAINING AND EVAL DOCS
t_docs = text_cleaning(train_docs)
e_docs = text_cleaning(eval_docs)

# COUNTS NUMBER OF WORDS IN DOC
train_docs_freqs = Counter()
for doc in train_docs:
    train_docs_freqs.update(doc)

# COUNTS HOW MANY POSITIVE VS NEGATIVE IN DOC
train_labels_freqs = Counter(train_labels)

# COUNTS NUMBER OF WORDS IN EVAL
eval_docs_freqs = Counter()
for doc in eval_docs:
    eval_docs_freqs.update(doc)

# COUNTS NUMBER OF POS VS NEG IN EVAL
eval_labels_freqs = Counter(eval_labels)

# PLOTS GRAPH OF POS VSS NEG FOR TRAINING SET
labels, values = zip(*train_labels_freqs.items())
indexes = np.arange(len(labels))
width = 0.2
plt.title("Train Labels Freq")
plt.ylabel("Frequency")
plt.xlabel("Sentiment polarity label ")
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()

# PLOTS GRAPH OF POS VS NEGATIVE IN EVAL
labels, values = zip(*eval_labels_freqs.items())
indexes = np.arange(len(labels))
width = 0.2
plt.title("Eval Labels Freq")
plt.ylabel("Frequency")
plt.xlabel("Sentiment polarity label ")
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()

# PLOTS GRAPH OF POS VS NEG IN ENTIRE DOC
labels, values = zip(*Counter(all_labels).items())
indexes = np.arange(len(labels))
width = 0.2
plt.title("All Docs Labels Freq")
plt.ylabel("Frequency")
plt.xlabel("Sentiment polarity label ")
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()

# ATTRIBUTES A NUMBER TO EACH WORD (TRAINING)
# BEFORE TEXT CLEANING (USED TO COMPARE TO TEXT CLEANING RESULT)
# vectorized_training = CountVectorizer().fit(d2array_to_string(train_docs))
# vectorized_training.vocabulary_

# ATTRIBUTES A NUMBER TO EACH WORD (TRAINING)
# ON CLEANED SET
vectorized_training = CountVectorizer().fit(t_docs)
vectorized_training.vocabulary_

# VECTORIZED WORDS (row, word#) BASED ON FREQUENCY IN LINE
# BEFORE CLEANING
# training_vector = vectorized_training.transform(d2array_to_string(train_docs))
# AFTER CLEANING
training_vector = vectorized_training.transform(t_docs)

# ATTRIBUTE SIGNIFICANCE RATION TO WORDS USING TfidTransformer FUNCTION BASED ON FREQUENCY OF THE WORDS IN LINE
# TRAINING
tfidf_transformer_training = TfidfTransformer().fit(training_vector)
significance_training = tfidf_transformer_training.transform(training_vector)

# Training set used for predictions
# CREATE MODEL TO BE USED IN PREDICTIONS
nbclf = MultinomialNB().fit(significance_training, train_labels)

# VECTORIZE EVAL SET (ASSIGN NUMBER TO WORD)
# BEFORE CLEANING
# vectorized_eval = CountVectorizer(vocabulary=vectorized_training.vocabulary_).fit(d2array_to_string(eval_docs))
# vectorized_eval.vocabulary_
# AFTER CLEANING
vectorized_eval = CountVectorizer(vocabulary=vectorized_training.vocabulary_).fit(e_docs)
vectorized_eval.vocabulary_

# BEFORE CLEANING
# eval_vector = vectorized_eval.transform(d2array_to_string(eval_docs))
# AFTER CLEANING
eval_vector = vectorized_eval.transform(e_docs)

# In[386]:
X_eval = eval_vector.toarray()
print(X_eval)

# In[388]:
tfidf_transformer_eval = TfidfTransformer().fit(eval_vector)
print(tfidf_transformer_eval)

# ASSIGN SIGNIFICANCE TO EVAL_VECTOR
significance_eval = tfidf_transformer_eval.transform(eval_vector)

# USE MODEL CREATED PREVIOUSLY TO PREDICT POS/NEG on eval set.
nbclf_predictions = nbclf.predict(significance_eval)
print(nbclf_predictions)

# CONFUSION MATRIX
# [ [A,B], --> first row outputs number of positive results for testing set
#   [C,D]] --> second row outputs number of negative results for testing set
# A --> our algorithm returned this many positive results CORRECTLY
# B --> our algorithm returned this many positive results incorrectly
# C -->  our algorithm returned this many Negative results incorrectly
# D --> our algorithm returned this many positive results CORRECTLY

# print("NAIVE BAYES CONFUSION MATRIX: ")
# print(confusion_matrix(eval_labels, nbclf_predictions))
#
# print(" NAIVE BAYES : Classification Report: ")
# print(classification_report(eval_labels, nbclf_predictions))

# TASK 2B)
# BASE DECISION TREE
# CREATE BASE DECISION TREE CLASSIFIER (CRIT = ENTROPY)
base_dtclf = DecisionTreeClassifier(criterion="entropy")

# FIT TRAINING SIGNIFICANCE AND TRAINING LABELS
base_dtclf.fit(significance_training, train_labels)

# USE BASE DECISION TREE CLASSIFIER TO PREDICT USING EVAL SIGNIFICANCE
base_dtclf_prediction = base_dtclf.predict(significance_eval)

# PRINT CONFUSION MATRIX
# print("BASE DT CONFUSION MATRIX: ")
# print(confusion_matrix(eval_labels, base_dtclf_prediction))
# [ [A,B], --> first row outputs number of positive results for testing set
#   [C,D]] --> second row outputs number of negative results for testing set
# A --> our algorithm returned this many positive results CORRECTLY
# B --> our algorithm returned this many positive results incorrectly
# C -->  our algorithm returned this many Negative results incorrectly
# D --> our algorithm returned this many positive results CORRECTLY

# print("BASE DT : Classification Report: ")
# print(classification_report(eval_labels, base_dtclf_prediction))

# TASK 2C) BEST DICISION TREE CLASSIFIER
best_dtclf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3000)
best_dtclf.fit(significance_training, train_labels)
best_dtclf_prediction = best_dtclf.predict(significance_eval)

# print("BEST DT CONFUSION MATRIX: ")
# print(confusion_matrix(eval_labels, best_dtclf_prediction))
#
# print("Best DT : Classification Report: ")
# print(classification_report(eval_labels, best_dtclf_prediction))

# ACCURACY SCORES
nb_accuracy_score = accuracy_score(eval_labels, nbclf_predictions)
base_accuracy_score = accuracy_score(eval_labels, base_dtclf_prediction)
best_accuracy_score = accuracy_score(eval_labels, best_dtclf_prediction)

# PRECISION SCORE
# MICRO = Calculate metrics globally by counting the total true positives, false negatives and false positives.
nb_precision_score = precision_score(eval_labels, nbclf_predictions, average="micro")
base_precision_score = precision_score(eval_labels, base_dtclf_prediction, average="micro")
best_precision_score = precision_score(eval_labels, best_dtclf_prediction, average="micro")

# RECALL
# MICRO = Calculate metrics globally by counting the total true positives, false negatives and false positives.
nb_recall_score = recall_score(eval_labels, nbclf_predictions, average="micro")
base_recall_score = recall_score(eval_labels, base_dtclf_prediction, average="micro")
best_recall_score = recall_score(eval_labels, best_dtclf_prediction, average="micro")

# F1
nb_f1_score = f1_score(eval_labels, nbclf_predictions, average="micro")
base_f1_score = f1_score(eval_labels, base_dtclf_prediction, average="micro")
best_f1_score = f1_score(eval_labels, best_dtclf_prediction, average="micro")

print("NB ACCURACY SCORE: ")
print(nb_accuracy_score)

print("BASE ACCURACY SCORE: ")
print(base_accuracy_score)

print("BEST ACCURACY SCORE: ")
print(best_accuracy_score)


# USING GRID SEARCH TO FIND BEST PARAMETERS TO USE
# def dtree_grid_search(X,y):
#     #create a dictionary of all values we want to test
#     param_grid = {
#         "criterion": ["entropy"],
#         "max_depth": [2000, 2500, 3000]
#     }
#     # decision tree model
#     dtree_model=DecisionTreeClassifier()
#     #use gridsearch to test all values
#     dtree_gscv = GridSearchCV(dtree_model, param_grid)
#     #fit model to data
#     dtree_gscv.fit(X, y)
#     return dtree_gscv.bestparams
#
# print(dtree_grid_search(significance_training, train_labels))

# TASK 4 DETERMINING WHY FALSE PREDICTIONS OCCURRED
# INCORRECT PREDICTIONS INDEX FUNCTION
def false_predictions(predictions, eval):
    false_predictions = []
    for i in range(len(eval_labels)):
        if predictions[i] != eval[i]:
            false_predictions.append(i)
    return false_predictions


def print_false_predictions(false_predictions):
    print_false_predictions = []
    for j in range(len(eval_docs)):
        if j in false_predictions:
            print_false_predictions.append(eval_docs[j])
    return d2array_to_string(print_false_predictions)


Testing = false_predictions(nbclf_predictions, eval_labels)
# print(Testing)
# print(len(Testing))

test_docs = print_false_predictions(Testing)

# PRINT NAIVE BAYES CONFUSION MATRIX, CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE) TO OUTPUT FILE
with open("Naive_Bayes_Classifier-all_sentiment_shuffled.txt", mode="w") as file:
    # CONFUSION MATRIX
    NBConfusionMatrix = confusion_matrix(eval_labels, nbclf_predictions)
    print("Naive Bayes Confusion Matrix:", file=file)
    print("--------------------------------", file=file)
    print(NBConfusionMatrix, file=file)
    # CLASSIFICATION REPORT
    print("\n Naive Bayes Classification Report", file=file)
    print("--------------------------------", file=file)
    print(classification_report(eval_labels, nbclf_predictions), file=file)
    # PRINT ROW NUMBER OF INSTANCE , INDEX NUMBER OF PREDICTED CLASS
    print("Row Number of instance, Index of predicted class", file=file)
    print("--------------------------------", file=file)
    counter = 0
    for lines in eval_docs:
        print(f'{split_point + counter}, {nbclf_predictions[counter]}', file=file)
        counter = counter + 1

# PRINT BASE-DT CONFUSION MATRIX, CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE) TO OUTPUT FILE
with open("Base_DT-all_sentiment_shuffled.txt", mode="w") as file:
    # CONFUSION MATRIX
    BaseDTConfusionMatrix = confusion_matrix(eval_labels, base_dtclf_prediction)
    print("Base DT Confusion Matrix:", file=file)
    print("--------------------------------", file=file)
    print(BaseDTConfusionMatrix, file=file)
    # CLASSIFICATION REPORT
    print("\n Base DT Classification Report", file=file)
    print("--------------------------------", file=file)
    print(classification_report(eval_labels, base_dtclf_prediction), file=file)
    # PRINT ROW NUMBER OF INSTANCE , INDEX NUMBER OF PREDICTED CLASS
    print("Row Number of instance, Index of predicted class", file=file)
    print("--------------------------------", file=file)
    counter = 0
    for lines in eval_docs:
        print(f'{split_point + counter}, {base_dtclf_prediction[counter]}', file=file)
        counter = counter + 1

# PRINT BEST-DT CONFUSION MATRIX, CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE) TO OUTPUT FILE
with open("Best_DT-all_sentiment_shuffled.txt", mode="w") as file:
    # CONFUSION MATRIX
    BestDTConfusionMatrix = confusion_matrix(eval_labels, best_dtclf_prediction)
    print("Best DT Confusion Matrix:", file=file)
    print("--------------------------------", file=file)
    print(BestDTConfusionMatrix, file=file)
    # CLASSIFICATION REPORT
    print("\n Best DT Classification Report", file=file)
    print("--------------------------------", file=file)
    print(classification_report(eval_labels, best_dtclf_prediction), file=file)
    # PRINT ROW NUMBER OF INSTANCE , INDEX NUMBER OF PREDICTED CLASS
    print("Row Number of instance, Index of predicted class", file=file)
    print("--------------------------------", file=file)
    counter = 0
    for lines in eval_docs:
        print(f'{split_point + counter}, {best_dtclf_prediction[counter]}', file=file)
        counter = counter + 1
