"""
neural_retrieval.py

Created on Mon Apr 3 2023

@author: Lukas

This file contains all methods for neural text retrieval.
"""

# import packages

import pandas as pd
import numpy as np
import os
import torch
import random

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup

from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# import functions form other files in this project

import isear as isear


# functions for creating and training a neural retrieval model

def tokenize_dataset(corpus: list, tokenizer: str) -> tuple:
    """
    Tokenize a dataset and return the input ids and attention masks.

    Parameters
    ----------
    corpus : The corpus to tokenize. One text per element.

    tokenizer : The tokenizer to use.

    Returns
    -------
    input_ids : The input ids for the given corpus.

    attention_masks : The attention masks for the given corpus.
    """
    assert tokenizer in ['bert'], "The given tokenizer is not supported."

    if tokenizer == 'bert':
        # load the BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

        # tokenize all texts in the corpus
        input_ids = []
        attention_masks = []

        for text in corpus:
            encoded_dict = tokenizer.encode_plus(
                                text,
                                add_special_tokens=True,
                                max_length=512,
                                pad_to_max_length=True,
                                return_attention_mask=True,
                                return_tensors='pt',
                           )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

        # convert the lists into tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


def create_dataset(input_ids: torch.Tensor, attention_masks: torch.Tensor, labels: torch.Tensor) -> TensorDataset:
    """
    Create a TensorDataset from the given input ids, attention masks, and labels.

    Parameters
    ----------
    input_ids : The input ids for the given corpus.

    attention_masks : The attention masks for the given corpus.

    labels : The labels for the given corpus.

    Returns
    -------
    dataset : The TensorDataset for the given corpus.
    """
    # combine the input ids and attention masks into a tensor dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset


def create_dataloaders(dataset: TensorDataset, batch_size: int) -> tuple:
    """
    Create dataloaders for the given dataset.

    Parameters
    ----------
    dataset : The TensorDataset for the given corpus.

    batch_size : The batch size to use.

    Returns
    -------
    train_dataloader : The dataloader for the training set.

    validation_dataloader : The dataloader for the validation set.
    """
    # calculate the number of samples to include in each set
    train_size = int(0.9 * len(dataset))
    validation_size = len(dataset) - train_size

    # divide the dataset by randomly selecting samples
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])

    # create the dataloaders for the training and validation sets
    train_dataloader = DataLoader(
                train_dataset,
                sampler=RandomSampler(train_dataset),
                batch_size=batch_size
            )

    validation_dataloader = DataLoader(
                validation_dataset,
                sampler=SequentialSampler(validation_dataset),
                batch_size=batch_size
            )

    return train_dataloader, validation_dataloader


def create_model(model_name: str, num_labels: int) -> BertForSequenceClassification:
    """
    Create a model for the given model name and number of labels.

    Parameters
    ----------
    model_name : The name of the model to create.

    num_labels : The number of labels to use.

    Returns
    -------
    model : The model for the given model name and number of labels.
    """
    assert model_name in ['bert'], "The given model name is not supported."

    if model_name == 'bert':
        # load the BERT model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-cased",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )

    return model


def train_bert(model: BertForSequenceClassification, train_dataloader: DataLoader, validation_dataloader: DataLoader,
                epochs: int) -> BertForSequenceClassification:
    """
    Train a model.

    Parameters
    ----------
    model : The model to train.

    train_dataloader : The dataloader for the training set.

    validation_dataloader : The dataloader for the validation set.

    epochs : The number of epochs to train the model for.

    Returns
    -------
    model : The trained model.
    """
    # get the device to train the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # send the model to the device
    model.to(device)

    # get the total number of training steps
    total_steps = len(train_dataloader) * epochs

    # create the optimizer
    optimizer = AdamW(model.parameters(), lr=5e-6, eps=1e-8)

    # create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=3, num_training_steps=total_steps)

    # set the seed for reproducibility
    seed_val = 60
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    loss_values = []

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            # remove the batch from the GPU to free up memory
            del b_input_ids, b_input_mask, b_labels

        avg_train_loss = total_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("Training complete!")

    return model  


def flat_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate the accuracy of the model.

    Parameters
    ----------
    preds : The predictions of the model.

    labels : The actual labels.

    Returns
    -------
    accuracy : The accuracy of the model.
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def run_inference(model, input_ids: torch.Tensor, attention_masks: torch.Tensor) -> torch.Tensor:
    """
    Run inference on the given model.

    Parameters
    ----------
    model : The model to run inference on.

    input_ids : The input ids to use.

    attention_masks : The attention masks to use.

    Returns
    -------
    predictions : The predictions of the model.
    """
    # get the device to run inference on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # send the model to the device
    model.to(device)

    # put the model in evaluation mode
    model.eval()

    # create the dataloader
    dataloader = DataLoader(
        TensorDataset(input_ids, attention_masks),
        batch_size=32,
        sampler = SequentialSampler(TensorDataset(input_ids, attention_masks))
        )
    
    # get the predictions
    predictions = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs[0]

        logits = logits.detach().cpu().numpy()

        predictions.append(logits)

        # remove the batch from the GPU
        del b_input_ids, b_input_mask

    # concatenate the predictions
    predictions = np.concatenate(predictions, axis=0)

    return np.argmax(predictions, axis=1).flatten()


def neural_retrieval(model_output: torch.Tensor) -> list:
    """
    A function that retrieves documents based on the model output.

    Parameters
    ----------
    model_output : The labels predicted by the model.

    Returns
    -------
    documents : The indices of the documents retrieved based on the keywords.
    """
    documents = []
    for i, label in enumerate(model_output):
        if label == 1:
            documents.append(i)

    return documents


def latent_representation(model: BertForSequenceClassification, input_ids: torch.Tensor,
                          attention_masks: torch.Tensor) -> torch.Tensor:
    """
    A function that retrieves the latent representations of a BERT model
    given the input ids and attention masks of the documents.

    Parameters
    ----------
    model : The model to retrieve the latent representation from.

    input_ids : The input ids to use.

    attention_masks : The attention masks to use.

    Returns
    -------
    latent_representation : The latent representation of the documents.
    """
    # get the device to run inference on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # send the model to the device
    model.to(device)

    # put the model in evaluation mode
    model.eval()

    # create the dataloader
    dataloader = DataLoader(
        TensorDataset(input_ids, attention_masks),
        batch_size=32,
        sampler = SequentialSampler(TensorDataset(input_ids, attention_masks))
        )
    
    # get the 768 dimensional latent representations before the classification layer
    latent_representations = []

    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask = batch

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        latent_representation = outputs[0]

        latent_representation = latent_representation.detach().cpu().numpy()

        latent_representations.append(latent_representation)

        # remove the batch from the GPU
        del b_input_ids, b_input_mask

    # concatenate the latent representations
    latent_representations = np.concatenate(latent_representations, axis=0)
    
    return latent_representations
