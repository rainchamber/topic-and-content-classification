"""
h-divergence.py

Created on Fri Mar 31 2023

@author: Lukas

This file contains all methods for computing the h-divergence.
"""

# import packages

import numpy as np
import torch, torchvision


# compute the H-divergence between the labeled data and the unlabeled data

def compute_H_divergence(labeled: np.ndarray, unlabeled: np.ndarray,
                         model: torch.nn.Sequential) -> float:
    """
    A function that computes the H-divergence between the labeled and unlabeled data.

    Parameters
    ----------
    labeled : The labeled data.

    unlabeled : The unlabeled data.

    model : The discriminative model.

    Returns
    -------
    H_divergence : The H-divergence between the labeled and unlabeled data.
    """
    # convert the data to torch tensors
    labeled = torch.from_numpy(labeled).float()
    unlabeled = torch.from_numpy(unlabeled).float()

    # for each element in the labeled data, compute the probabilities of the classes
    p_L = model(labeled)
    p_U = model(unlabeled)

    # sum the probabilities of class 0 for each element in the labeled data and divide by the number of elements
    p_L_0 = torch.sum(p_L[:, 0])
    p_L_0 /= labeled.shape[0]

    # sum the probabilities of class 0 for each element in the unlabeled data and divide by the number of elements
    p_U_0 = torch.sum(p_U[:, 0])
    p_U_0 /= unlabeled.shape[0]

    # compute the H-divergence as the absolute difference between p_U_0 and p_L_0
    H_divergence = torch.abs(p_U_0 - p_L_0)

    # convert the H-divergence to a numpy float
    return H_divergence.item()


# train a discriminative model on the labeled data and unlabeled data

def train_discriminative_model(labeled: np.ndarray, unlabeled: np.ndarray,
                               input_shape: int) -> torch.nn.Sequential:
    """
    A function that trains and returns a discriminative model on the labeled and unlabeled data.

    Parameters
    ----------
    labeled : The labeled data.

    unlabeled : The unlabeled data.

    input_shape : The number of features in the dataset.

    Returns
    -------
    model : The trained discriminative model.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0], 1), dtype='int')
    y_U = np.ones((unlabeled.shape[0], 1), dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    X_train = torch.from_numpy(X_train).float()
    Y_train = torch.from_numpy(Y_train).squeeze().long()

    # build the model:
    model = get_discriminative_model(input_shape)

    # train the model using torch:
    batch_size = 100
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i in range(0, X_train.shape[0], batch_size):
            x = X_train[i:i + batch_size]
            y = Y_train[i:i + batch_size]
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

    return model


# we use a 3-layer MLP as the discriminative model

def get_discriminative_model(input_shape: int) -> torch.nn.Sequential:
    """
    The MLP model for discriminative active learning, without any regularization techniques.

    Parameters
    ----------
    input_shape : The number of features in the dataset.

    Returns
    -------
    model : The MLP model.
    """
    width = input_shape
    model = torch.nn.Sequential(
        torch.nn.Linear(width, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 2),
        torch.nn.Softmax(dim=1)
    )

    return model


# get the latent representation of the data using the trained model

def get_latent_representation(model: torch.nn.Sequential, X: np.ndarray) -> np.ndarray:
    """
    A function that computes the latent representation of the data using the trained model.

    Parameters
    ----------
    model : The trained model.

    X : The data.

    Returns
    -------
    latent_representation : The latent representation of the data.
    """
    X = torch.from_numpy(X).float()
    latent_representation = model(X)
    latent_representation = latent_representation.detach().numpy()

    return latent_representation