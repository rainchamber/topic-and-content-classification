"""
discriminative_active_learning.py

Created on Fri Mar 31 2023

@author: Lukas

This file contains all methods for discriminative active learning.
Note that here, we only choose the training dataset, we do not train the model.
"""

# import packages

import numpy as np
import torch, torchvision


# import functions form other files in this project

import batch_functions as bf
import h_divergence as hd


# functions for discriminative active learning

def discriminative_active_learning(Model: torch.nn.Sequential, X_train: np.ndarray, labeled_idx: np.ndarray, 
                                   get_latent_rep, batch_size: int, n_batches: int) -> np.ndarray:
    """
    Perform discriminative active learning for a given model and dataset.

    Parameters
    ----------
    Model : The model to perform active learning for.

    X_train : The training data.

    labeled_idx : The indices of the labeled data.

    get_latent_rep : The function to get the latent representation of the data.

    batch_size : The size of the active learning batches.

    n_batches : The number of batches to select.

    Returns
    -------
    batches : The batches of indices to label.
    """
    unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

    # get the latent representation of X_train using get_latent_representation
    latent_representation = get_latent_rep(Model, X_train) 

    # iteratively sub-sample the unlabeled data using the DAL routine
    batches = []
    for i in range(n_batches):

        # train a discriminative model on the labeled data and unlabeled data
        discriminative_model = hd.train_discriminative_model(X_train[labeled_idx],
                                                             latent_representation[unlabeled_idx],
                                                             latent_representation.shape[1])
        
        # choose the next batch of indices to label using the discriminative model
        batch = choose_batch(discriminative_model, X_train[unlabeled_idx],
                                latent_representation[unlabeled_idx], batch_size)
        
        # add the batch to the list of batches
        batches.append(batch)

        # update the labeled and unlabeled indices
        labeled_idx = np.concatenate((labeled_idx, unlabeled_idx[batch]))
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

    return batches


def get_unlabeled_idx(X_train: np.ndarray, labeled_idx: np.ndarray) -> np.ndarray:
    """
    Get the indices of the unlabeled data.

    Parameters
    ----------
    X_train : The training data.

    labeled_idx : The indices of the labeled data.

    Returns
    -------
    unlabeled_idx : The indices of the unlabeled data.
    """
    unlabeled_idx = np.array([i for i in range(X_train.shape[0]) if i not in labeled_idx])

    return unlabeled_idx


def choose_batch(discriminative_model: torch.nn.Sequential, unlabeled_idx: np.ndarray,
                 latent_representation: np.ndarray, batch_size: int) -> np.ndarray:
    """
    Choose a batch of indices to label.

    Parameters
    ----------
    discriminative_model : The discriminative model.

    unlabeled_idx : The indices of the unlabeled data.

    latent_representation : The latent representation of the unlabeled data.

    batch_size : The size of the batch.

    Returns
    -------
    batch : The batch of indices to label.
    """
    # compute the H-divergence between the labeled data and the unlabeled data
    H_divergence = hd.compute_H_divergence(discriminative_model, latent_representation)

    # sort the H-divergence in descending order
    H_divergence_sorted = np.sort(H_divergence)[::-1]

    # get the indices of the H-divergence in descending order
    H_divergence_sorted_idx = np.argsort(H_divergence)[::-1]

    # get the indices of the unlabeled data in the order of the H-divergence
    unlabeled_idx_sorted = unlabeled_idx[H_divergence_sorted_idx]

    # get the batch of indices to label
    batch = unlabeled_idx_sorted[:batch_size]

    return batch