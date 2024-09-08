#!/usr/bin/env python

'''
# TODO: expand description
This script performs text classification using multiple machine learning algorithms (NB, DT, RF, KNN, SVC, LinearSVC).
'''

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='datasets/train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='datasets/dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-a", "--algorithm", default='naive_bayes', type=str,
                        help="Machine Learning Algorithm to use, options are: naive_bayes, decision_tree, random_forest, knn, svc, linear_svc")
    args = parser.parse_args()
    return args

def read_corpus(corpus_file, use_sentiment):
    '''
    This function reads the corpus file and converts the textual data into a format more suitable for classification tasks later on

    @param corpus_file: input file consisting of textual data
    @param use_sentiment: boolean indicating whether sentiment will need to be used
    @return: the documents (list of tokens) and
            labels (target labels for each document, this can be either sentiment labels or category labels)
    '''
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            documents.append(tokens[3:])
            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append(tokens[1])
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp

# Return the classifier indicated by the input arguments
def get_classifier(algorithm):
    '''
    This function reads the algorithm given in the input parameters and returns the corresponding classifier
    
    @param algorithm: name of the machine learning algorithm as indicated in the input parameters
    @return: the classifier corresponding to the inputted algorithm
    @raise ValueError: raises an exception when the inputted algorithm can not be matched to a classifier
    '''
    # Naive bayes implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
    if algorithm == 'naive_bayes':
        return MultinomialNB()
    # Decision tree implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    if algorithm == 'decision_tree':
        return DecisionTreeClassifier()
    # Decision tree implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    if algorithm == 'random_forest':
        return RandomForestClassifier()
    # K Nearest Neighbours implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    if algorithm == 'knn':
        return KNeighborsClassifier()
    # Support Vector Classification implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    if algorithm == 'svm':
        return SVC()
    # Linear Support Vector Classification implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    if algorithm == 'svm_linear':
        return LinearSVC()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

if __name__ == "__main__":
    # Parse the input arguments
    args = create_arg_parser()

    # Load the train and test datasets
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if args.tfidf:
        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
    else:
        # Bag of Words vectorizer
        vec = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # Get the classifier that was given in the input arguments
    chosen_classifier = get_classifier(args.algorithm)

    # Create a pipeline by combining the chosen vectorizer and classifier
    classifier = Pipeline([('vec', vec), ('cls', chosen_classifier)])

    # Train the model using the training set
    classifier.fit(X_train, Y_train)

    # Let the model make predictions on the test set
    Y_pred = classifier.predict(X_test)

    # Evaluate the predictions that were made by comparing them to the ground truth, apply several metrics from the sklearn library for this:
    # Accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    acc = accuracy_score(Y_test, Y_pred)
    print(f"Final accuracy: {acc}")

    # Precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    precision = precision_score(Y_test, Y_pred)
    print(f"Final precision score: {precision}")

    # Recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    recall = recall_score(Y_test, Y_pred)
    print(f"Final recall score: {recall}")

    # F1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    f1 = f1_score(Y_test, Y_pred)
    print(f"Final f1 score: {f1}")