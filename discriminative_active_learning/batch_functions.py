"""
batch_functions.py

Created on Fri Mar 31 2023

@author: Lukas

This file contains all methods for measuring the quality of a batch.
"""

# import packages

import numpy as np


# functions for measuring the quality of a batch

def get_diversity(dataset: np.ndarray, batch: np.ndarray) -> float:
    """
    Compute the diversity of a dataset.

    Parameters
    ----------
    dataset : The dataset to compute the diversity for.

    batch : The batch to compute the diversity for.

    Returns
    -------
    diversity : The diversity of the dataset.
    """
    minima = [get_min_distance_to_batch(point, batch) for point in dataset]
    diversity =  len(dataset) / sum(minima)

    return diversity
    

def get_min_distance_to_batch(point: np.ndarray, batch: np.ndarray) -> float:
    """
    Compute the minimum distance of a point to a batch.

    Parameters
    ----------
    point : The point to compute the minimum distance for.

    batch : The batch to compute the minimum distance for.

    Returns
    -------
    min_distance : The minimum distance of the point to the batch.
    """
    distances = [np.linalg.norm(point - other_point) for other_point in batch]
    
    return min(distances)


def get_representativeness(dataset: np.ndarray, batch: np.ndarray, k: int = 10) -> float:
    """
    Compute the representativeness of a dataset.

    Parameters
    ----------
    dataset : The dataset to compute the representativeness for.

    batch : The batch to compute the representativeness for.

    k : int, optional

    Returns
    -------
    representativeness : The representativeness of the dataset.
    """
    knn_densities = [compute_knn_density(point, dataset, k) for point in dataset]
    representativeness = len(dataset) / sum(knn_densities)

    return representativeness


def compute_knn_density(sample: np.ndarray, dataset: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the density of a point in a dataset.

    Parameters
    ----------
    sample : The point to compute the density for.

    dataset : The dataset to compute the density for.

    k : the parameter k for the k-nearest-neighbors algorithm.

    Returns
    -------
    density : The density of each point in the dataset.
    """
    distances = np.array([np.linalg.norm(sample - other_sample) for other_sample in dataset])
    k_nearest_neighbors = np.argpartition(distances, k)[:k]

    cosine = np.array([np.dot(sample, dataset[neighbor]) for neighbor in k_nearest_neighbors])
    density = np.sum(cosine) / k

    return density