# COMP472-Assignment1
COMP 472 - Assignment 1 - Experimenting with Machine Learning

## Description
This assignment uses **Python** and **Sckit-Learn Machine Learning Framework** to analyze a given data set.
The machine learning models used in this assignment are :
  - Naive Bayes Classifier
  - Base Decision Tree
  - Best Decision Tree


### Data Set  
Each line of the data set is formatted as follows:\
**[Category_Label] [Sentiment_Polarity_Label] [Document_Identifier] [Description]**\
```music neg 544.txt i was misled and thought i was buying the entire cd and it contains one song```

## Getting Started
**Step 1:** Open a new project on Pycharm (Make sure you have **Python Version 3.9** or later installed on your computer).\
**Step 2:** Navigate to Pycharm settings

       > (Mac OS: Pycharm -> Preferences -> Project -> Python Interpretor -> "+")
       
       > (Windows OS: File -> Settings -> Project -> Python Interpretor -> "+")
       
**Step 3:** Install the following packages: 
  - numpy
  - scikit-learn
  - matplotlib
  - scipy
  - nltk

**Step 4:** Import Dataset File Into Project

       > Drag dataset file into top level of project
       
**Step 5:** Change file path to use data set in code (*Line 44*)

all_docs, all_labels = read_documents(`"all_sentiment_shuffled.txt"`)


### Import Usage:
Imports are already in code:
```
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

from sklearn.tree import DecisionTreeClassifier

import numpy as np
import string
import matplotlib.pyplot as plt
```

## Overall code process and code snippets
**Code Process:\
Step 1: Extract the data and split it into a training set and a testing set**
```
all_docs, all_labels = read_documents("all_sentiment_shuffled.txt")
split_point = int(0.80 * len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]
```

**Step 2: Clean the text**
```
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
```

**Step 3: Plot the label distribution:**
```
labels, values = zip(*eval_labels_freqs.items())
indexes = np.arange(len(labels))
width = 0.2
plt.title("Eval Labels Freq")
plt.ylabel("Frequency")
plt.xlabel("Sentiment polarity label ")
plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, labels)
plt.show()
```

**Step 4: Vectorize the text using CountVectorizer in order to insert a valid input into the Classifiers:**
```
vectorized_training = CountVectorizer().fit(t_docs)
vectorized_training.vocabulary_
```

**Step 5: Attribute a significance ratio to each word in their respective sentence using TfidfTransformer:**
```
tfidf_transformer_training = TfidfTransformer().fit(training_vector)
significance_training = tfidf_transformer_training.transform(training_vector)
```

**Step 6: Run the classifiers on the processed training dataset:**
```
nb_classifier = MultinomialNB().fit(significance_training, train_labels)
base_dtclf = DecisionTreeClassifier(criterion="entropy").fit(significance_training, train_labels)
best_dtclf = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=3200).fit(significance_training, train_labels)
```

**Step 7: Predict the results for each classifier using the testing dataset:**
```
nb_predictions = model.predict(significance_eval)
base_dtclf_prediction = base_dtclf.predict(significance_eval)
best_dtclf_prediction = best_dtclf.predict(significance_eval)
```

**Step 8: Compute the accuracy for each prediction:**
```
nb_accuracy_score = accuracy_score(eval_labels, all_predictions)
nb_precision_score = precision_score(eval_labels, all_predictions, average="micro")
nb_recall_score = recall_score(eval_labels, all_predictions, average="micro")
nb_f1_score = f1_score(eval_labels, all_predictions, average="micro")
```

**Step 9: Present the results in a file for each classifier:**
```
with open("Naive_Bayes_Classifier-all_sentiment_shuffled.txt", mode="w") as file:
    NBConfusionMatrix = confusion_matrix(eval_labels, all_predictions)
    print("Naive Bayes Confusion Matrix:", file=file)
    print("--------------------------------", file=file)
    print(NBConfusionMatrix, file=file)
    print("\n Naive Bayes Classification Report", file=file)
    print("--------------------------------", file=file)
    print(classification_report(eval_labels, all_predictions), file=file)

    print("Row Number of instance, Index of predicted class", file=file)
    print("--------------------------------", file=file)
    counter = 0
    for lines in eval_docs:
        print(f'{split_point + counter}, {all_predictions[counter]}', file=file)
        counter = counter + 1
```

## Get Help
To get help or ask questions, Please Contact any of the following students: 
 - **Full Name:** Kamil Geagea\
   **Student ID:** 40052432\
   **Github Username:** kamilgeagea\
   **Email Address:** kamilgeagea8199@gmail.com
   
 - **Full Name:** Marjana Upama\
   **Student ID:** 40058393\
   **Github Username:** Marjanaupama\
   **Email Address:** zana.zinly@gmail.com
   
 - **Full Name:** JC Manikis\
   **Student ID:** 26884466\
   **Github Username:** jmanikis\
   **Email Address:** jmanikis@icloud.com
   
 - **Full Name:** Mair Elbaz\
   **Student ID:** 40004558\
   **Github Username:** mairsarmy32\
   **Email Address:** mairelbaz552@hotmail.com

## References
- [Scikit Multinomial Naive Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [Scikit Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Scikit Precision Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [Scikit Grid Search](https://scikit-learn.org/stable/modules/grid_search.html)
- [Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
