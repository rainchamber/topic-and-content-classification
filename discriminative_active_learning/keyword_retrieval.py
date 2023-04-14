"""
keyword_retrieval.py

Created on Mon Apr 3 2023

@author: Lukas

This file contains all methods for keyword-based text retrieval.
"""

# import packages

import numpy as np
import os
import pandas as pd

# functions for keyword-based text retrieval

def keyword_retrieval(keywords: list, corpus: pd.core.series.Series) -> list:
    """
    A function that retrieves documents based on keywords.

    Parameters
    ----------
    keywords : The keywords to retrieve documents for.

    corpus : The dataframe containing the texts.

    Returns
    -------
    documents : The indices of the documents retrieved based on the keywords.
    """
    documents = []

    # for each element in the corpus, check if at least one keyword is in the text
    for i, text in enumerate(corpus):
        if any([keyword in text for keyword in keywords]):
            documents.append(i)

    return documents