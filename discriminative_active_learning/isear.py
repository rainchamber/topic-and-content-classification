"""
isear.py

Created on Mon Apr 3 2023

@author: Lukas

This file contains all methods for working with the ISEAR dataset.
"""

# import packages

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
from scipy.stats.stats import pearsonr  

# functions for working with the ISEAR dataset

def load_isear() -> pd.core.frame.DataFrame:
    """
    Load the ISEAR dataset.

    Returns
    -------
    isear : The ISEAR dataset.
    """
    # mount the Google Drive
    from google.colab import drive
    drive.mount('/content/drive')

    # load the dataset
    isear = pd.read_csv("/content/drive/MyDrive/Topic and Content Classification/DATA.csv")

    return isear


def get_isear_positives(dataset: pd.core.frame.DataFrame, emotion: int) -> list:
    """
    Get the indices of the positive examples of a given emotion from the ISEAR dataset.

    Parameters
    ----------
    dataset : The ISEAR dataset.

    emotion : The emotion to get the positive examples for.

    Returns
    -------
    positives : The indices of the positive examples of the given emotion.
    """
    # get the positive examples of the given emotion
    positives = dataset[dataset['EMOT'] == emotion].index.tolist()

    return positives


def load_isear_texts(isear: pd.core.frame.DataFrame, int_low: int, int_high: int) -> list:
    """
    Load the texts of the ISEAR dataset.

    Parameters
    ----------
    isear : The ISEAR dataset.

    int_low : The lower bound of the interval of texts to load.

    int_high : The upper bound of the interval of texts to load.

    Returns
    -------
    texts : The texts of the ISEAR dataset.
    """

    # get the texts of the ISEAR dataset
    texts = [isear['SIT'][i] for i in range(int_low, int_high)]

    return texts


def load_binary_isear_labels(isear: pd.core.frame.DataFrame, int_low: int, int_high: int, emot: int) -> list:
    """
    Load the labels of the ISEAR dataset with binary labels,
    i.e. 0 if the emotion is not the given emotion and 1 if the emotion is the given emotion.

    Parameters
    ----------
    isear : The ISEAR dataset.

    int_low : The lower bound of the interval of labels to load.

    int_high : The upper bound of the interval of labels to load.

    emot : The emotion to get the labels for.

    Returns
    -------
    labels : The labels of the ISEAR dataset.
    """
    # get the labels of the ISEAR dataset
    labels = [1 if isear['EMOT'][i] == emot else 0 for i in range(int_low, int_high)]

    return labels


def get_correlation(isear: pd.core.frame.DataFrame, variable: str,
                    error_type: str, true_positives: list, retrieval: list) -> tuple:
    """
    Get the correlation between a variable and the retrieved documents.

    Parameters
    ----------
    isear : The ISEAR dataset.

    variable : The variable in the dataset to get the correlation for.

    error_type : The error type to get the correlation for.

    true_positives : The indices of the true positives.

    retrieval : The indices of the retrieved documents.

    Returns
    -------
    correlation : The correlation between the variable and the retrieved documents.
    """
    assert error_type in ['false_positive', 'false_negative', 'error'], 'Unknown error type.'

    # get the variable of the ISEAR dataset
    variable = isear[variable]

    # create a vector of zeroes with ones where the error type is the given error type
    retrieved_docs = np.zeros(len(variable))
    if error_type == 'false_positive':
        retrieved_docs[retrieval] = 1
        retrieved_docs[true_positives] = 0

    elif error_type == 'false_negative':
        retrieved_docs[true_positives] = 1
        retrieved_docs[retrieval] = 0

    # compare the retrieved documents with the correct documents
    elif error_type == 'error':
        retrieved_docs[retrieval] = 1
        retrieved_docs[true_positives] -= 1
        
        # take the absolute value of every element in the vector
        retrieved_docs = np.abs(retrieved_docs)

    # get the correlation between the variable vector and the retrieval vector
    correlation = pearsonr(variable, retrieved_docs)

    return correlation


def get_mutual_information(isear: pd.core.frame.DataFrame, variable: str, retrieval: list) -> float:
    """
    Get the mutual information between a variable and the retrieved documents.

    Parameters
    ----------
    isear : The ISEAR dataset.

    variable : The variable in the dataset to get the mutual information for.

    retrieval : The indices of the retrieved documents.

    Returns
    -------
    mutual_information : The mutual information between the variable and the retrieved documents.
    """
    # get the variable of the ISEAR dataset
    variable = isear[variable]

    # create a vector of zeroes with ones at the retrieved documents
    retrieved_docs = np.zeros(len(variable))
    retrieved_docs[retrieval] = 1

    # get the mutual information between the variable vector and the retrieval vector
    mutual_information = mutual_info_score(variable, retrieved_docs)

    return mutual_information[0], mutual_information[1]