#!/usr/bin/env python

'''
# TODO: expand description
This script performs text classification using multiple machine learning algorithms (NB, DT, RF, KNN, SVC, LinearSVC).

@arg -t, --train_file: specifies the train file to learn from
@arg -d, --dev_file: specifies the dev file to evaluate on
@arg -s, --sentiment: specifies whether sentiment analysis is performed
@arg -tf, --tfidf: specifies whether to use TfidfVectorizer
@arg -a, --algorithm: specifies the Machine Learning algorithm to use
@arg -avg, --average: specifies the averaging technique used in evaluation
'''
'''
This script contains the following changes: 
1. It uses lemmatisation instead of stemming
2. It incorporates character and word N-grams
Hongxu Zhou
10/Sept/2024
'''
import re

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion # New function FU added
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import nltk
from nltk.stem import WordNetLemmatizer # Since we use this lemmatiser, consider adding POS in the next test.


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='/Users/hongxuzhou/LfD/Week1/Learning-From-Data-Week1/train.txt', type=str,
                        help="Train file to learn from (default train.txt)")
    parser.add_argument("-d", "--dev_file", default='/Users/hongxuzhou/LfD/Week1/Learning-From-Data-Week1/dev.txt', type=str,
                        help="Dev file to evaluate on (default dev.txt)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-a", "--algorithm", default='naive_bayes', type=str,
                        help="Machine Learning Algorithm to use. Options are: naive_bayes, decision_tree, random_forest, knn, svc, linear_svc")
    parser.add_argument("-avg", "--average", default='weighted', type=str,
                        help="Averaging technique to use in evaluation. Options are: binary, micro, macro, weighted, samples")
    
    # The new added character ngram arg
    parser.add_argument("-cn", "--char_ngram", type = int, default = 0,
                        help = "Use character n-gram. Please specify the maximum n-gram size") 
    # The new added word ngram arg
    parser.add_argument("-wn", "--word_ngram", type = int, default = 1,
                        help = "Use word n-gram. Please specify the maximum n-gram size")
    
    # The new added lemmatisation arg
    parser.add_argument("-l", "--lemmatize", action = "store_true",
                        help = "Use lemmatization for text preprocessing")
    args = parser.parse_args()
    
    return args

'''
def read_corpus(corpus_file, use_sentiment):
    
    This function reads the corpus file and converts the textual data into a format more suitable for classification tasks later on

    @param corpus_file: input file consisting of textual data
    @param use_sentiment: boolean indicating whether sentiment will need to be used
    @return: the documents (list of tokens) and
            labels (target labels for each document, this can be either sentiment labels or category labels)
    
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
'''

    # The new corpus reading function written by Claude, to slove the "list" object has no attribute "lower"
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            parts = line.strip().split(maxsplit=3)
            if len(parts) < 4:
                continue  # Skip lines that don't have enough parts
            category, sentiment, _, text = parts
            if text:  # Only add non-empty documents
                documents.append(text)
                if use_sentiment:
                    labels.append(sentiment)
                else:
                    labels.append(category)
    return documents, labels

def identity(inp):
    '''Dummy function that just returns the input'''
    return inp

# Initialise the WordNet Lemmatizer 
Lemmatizer = WordNetLemmatizer()

# Construct the function of lemmatization
def lemma_process(text, lemmatize=False):
    '''
    1. lowercasing the text
    2. removing special characters & digits
    3. lemmatising
    '''
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Lowercasing convert 
    text = text.lower()
    
    # 2. Remove all the non-alphabetical letters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Lemmatizer apply
    if lemmatize:
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(lemmatized_words)
    
    return text

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
    if algorithm in ['svc','svm']:
        return SVC()
    # Linear Support Vector Classification implementation from sklearn
    # sklearn documentation can be found at: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    if algorithm in ['svm_linear','svc_linear']:
        return LinearSVC()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

if __name__ == "__main__":
    # Parse the input arguments
    args = create_arg_parser()
    '''
    # Import training and testing datasets
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment) # Didn't get this part, will look after it later.
    
    #Preprocess the data by using the function above
    
    X_train = [lemma_process(doc, args.lemmatize) for doc in X_train]
    X_test = [lemma_process(doc, args.lemmatize) for doc in X_test] # Also lost in this part too.
    #The two lines above are corrected by Claude, and I dont know where it went wrong, the next version is:
    
    X_train = [lemma_process(' '.join(doc), args.lemmatize) for doc in X_train]
    X_test = [lemma_process(' '.join(doc), args.lemmatize) for doc in X_test]
    '''
    X_train, Y_train = read_corpus(args.train_file, args.sentiment)
    X_test, Y_test = read_corpus(args.dev_file, args.sentiment)

    X_train_processed = [lemma_process(doc, args.lemmatize) for doc in X_train]
    X_test_processed = [lemma_process(doc, args.lemmatize) for doc in X_test]

    # Filter out empty documents
    X_train_filtered = []
    Y_train_filtered = []
    for x, y in zip(X_train_processed, Y_train):
        if x:  # Only keep non-empty documents
            X_train_filtered.append(x)
            Y_train_filtered.append(y)

    X_test_filtered = []
    Y_test_filtered = []
    for x, y in zip(X_test_processed, Y_test):
        if x:  # Only keep non-empty documents
            X_test_filtered.append(x)
            Y_test_filtered.append(y)

    # Use the filtered data for further processing
    # Create feature extractors
    vectorizer_class = TfidfVectorizer if args.tfidf else CountVectorizer
    
    # ... (rest of the code remains the same, but use X_train_filtered, Y_train_filtered, X_test_filtered, Y_test_filtered)
    
    # Create a word-level n-gram extractor
    word_ngram = vectorizer_class(ngram_range = (1, args.word_ngram))
    
    feature_extractor = [('word', word_ngram)]
    
    # Add a character-level n-gram feature extracter which can be used on request.
    if args.char_ngram > 0:
        char_ngram = vectorizer_class(analyzer = "char", ngram_range = (1, args.char_ngram)) # What is the length of the Char ngram? should be at least 3 and 5
        feature_extractor.append(('char', char_ngram)) # why here are two layers of brackets
    # Combine features using FeatureUnion
    combined_features = FeatureUnion(feature_extractor) # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
    
    # Get the specified classifier
    chosen_classifier = get_classifier(args.algorithm)
    chosen_average = args.average
    
    #Create a pipeline that combines feature extraction and classification
    classifier = Pipeline([
        ('features', combined_features),
        ('cls', chosen_classifier)
    ])
    
    # Train the model
    classifier.fit(X_train, Y_train)
    
    # Make predictions on the test set
    Y_pred = classifier.predict(X_test)
    
    '''
    # Feature Implementation Adjustments start here
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

    # Get the averaging method for multi-class classification
    chosen_average = args.average

    # Create a pipeline by combining the chosen vectorizer and classifier
    classifier = Pipeline([('vec', vec), ('cls', chosen_classifier)])
'''

    # General metrics

    # Evaluate the predictions that were made by comparing them to the ground truth, apply several metrics from the sklearn library for this:
    # Accuracy: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    acc = accuracy_score(Y_test, Y_pred)
    print(f"General accuracy: {acc}")

    # Precision: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    precision = precision_score(Y_test, Y_pred, average=chosen_average)
    print(f"General precision score: {precision}")

    # Recall: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    recall = recall_score(Y_test, Y_pred, average=chosen_average)
    print(f"General recall score: {recall}")

    # F1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    f1 = f1_score(Y_test, Y_pred, average=chosen_average)
    print(f"General f1 score: {f1}\n")
    

    # Confusion matrix: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    confusion = confusion_matrix(Y_test, Y_pred)
    print(f"Confusion matrix:\n{confusion}")  


    print("\nPer-class scores")

    #Classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    class_rep = classification_report(Y_test,Y_pred)
    print(class_rep)