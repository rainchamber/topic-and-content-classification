"""
evaluate_retrieval.py

Created on Mon Apr 3 2023

@author: Lukas

This file contains all methods for evaluating the quality of a retrieval.
"""

# import packages

import numpy as np

# functions for evaluating the quality of a retrieval

def compute_precision(correct_documents: list, retrieved_documents: list) -> float:
    """
    Given a list of correct documents and a list of retrieved documents, compute the precision.

    Parameters
    ----------
    correct_documents : The list of correct documents.

    retrieved_documents : The list of retrieved documents.

    Returns
    -------
    precision : The precision of the retrieval.
    """
    # compute the number of retrieved documents
    n_retrieved = len(retrieved_documents)

    # compute the number of correct and retrieved documents
    n_correct_and_retrieved = len([doc for doc in retrieved_documents if doc in correct_documents])

    # compute the precision
    precision = n_correct_and_retrieved / n_retrieved

    return precision


def compute_recall(correct_documents: list, retrieved_documents: list) -> float:
    """
    Given a list of correct documents and a list of retrieved documents, compute the recall.

    Parameters
    ----------
    correct_documents : The list of correct documents.

    retrieved_documents : The list of retrieved documents.

    Returns
    -------
    recall : The recall of the retrieval.
    """
    # compute the number of correct documents
    n_correct = len(correct_documents)

    # compute the number of correct and retrieved documents
    n_correct_and_retrieved = len([doc for doc in retrieved_documents if doc in correct_documents])

    # compute the recall
    recall = n_correct_and_retrieved / n_correct

    return recall


def compute_f1_score(correct_documents: list, retrieved_documents: list) -> float:
    """
    Given a list of correct documents and a list of retrieved documents, compute the F1 score.

    Parameters
    ----------
    correct_documents : The list of correct documents.

    retrieved_documents : The list of retrieved documents.

    Returns
    -------
    f1_score : The F1 score of the retrieval.
    """
    # compute the precision
    precision = compute_precision(correct_documents, retrieved_documents)

    # compute the recall
    recall = compute_recall(correct_documents, retrieved_documents)

    # compute the F1 score
    f1_score = 2 * precision * recall / (precision + recall)

    return f1_score