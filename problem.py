import os
import sys
import pickle
import numpy as np
from sklearn.model_selection import GroupKFold
import re
from collections import Counter
import rampwf as rw
from sklearn.model_selection import KFold


import sys
import os

# Get the absolute path of the directory containing ramp_custom
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from ramp_custom.workflow_ConditionedImageGenerator import ConditionedImageGenerator
from ramp_custom.prepare_data import prepare_data

class utils:
    """Utility functions helpful in the challenge."""

    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from ramp_custom import (FID,)
    from ramp_custom import PredictionType_generation as ptg


problem_title = "Text-Conditioned Image Generation for Interior Design"

Predictions = utils.ptg.Generation
workflow = ConditionedImageGenerator()
score_type_2 = utils.FID.FID(precision=3)

score_types = [
    utils.FID.FID(precision=3),
    #utils.clip_score.CLIPScore(precision=3),
]


"""
Problem definition file for the custom challenge.

This file implements:
  - I/O methods (get_train_data and get_test_data) to load training and test data.
    It uses the prepare_data module to obtain a DataLoader, converts the obtained
    text embeddings and images (in tensor form) to NumPy arrays, and applies a quick-test
    mode if specified.
  - A cross-validation scheme (get_cv) to split the training data.
"""


def _get_data_from_prepare_data(text_file, image_dir):
    """
    Load data using the prepare_data module.

    This function obtains a DataLoader, iterates over its batches converting
    text embeddings and image tensors to NumPy arrays, and concatenates the results.

    Parameters
    ----------
    text_file : str
        Path to the text file containing descriptions.
    image_dir : str
        Directory containing the images.

    Returns
    -------
    X : numpy.ndarray
        Array of text embeddings.
    y : numpy.ndarray
        Array of images.
    """
    dataloader = prepare_data(text_file, image_dir)
    text_embeddings_list = []
    image_list = []

    for text_embedding, image_tensor in dataloader:
        # Convert PyTorch tensors to NumPy arrays.
        text_embeddings_list.append(text_embedding.numpy())
        image_list.append(image_tensor.numpy())

    # Concatenate all batches along the first (batch) dimension.
    X = np.concatenate(text_embeddings_list, axis=0)
    y = np.concatenate(image_list, axis=0)
    return X, y


def _get_data(path=".", split="train"):
    """
    Get the data for the given split (train or test).

    Parameters
    ----------
    path : str, optional
        Base path to the data directory. Default is the current directory.
    split : str, optional
        The data split to retrieve, either "train" or "test". Default is "train".

    Returns
    -------
    X : numpy.ndarray
        Array of text embeddings.
    y : numpy.ndarray
        Array of images.
    """
    if split == "train":
        text_file = os.path.join(path, "data\public", "train\captions.txt")
        image_dir = os.path.join(path, "data\public", "train")
    elif split == "test":
        text_file = os.path.join(path, "data\public", "test\captions.txt")
        image_dir = os.path.join(path, "data\public", "test")
    else:
        raise ValueError("split must be either 'train' or 'test'")

    X, y = _get_data_from_prepare_data(text_file, image_dir)

    # If quick-test mode is enabled, only select a small subset of the data.
    if os.environ.get("RAMP_TEST_MODE", False):
        # For example, only use the first 20 samples.
        X = X[:10]
        y = y[:10]

    return X, y


def get_train_data(path="."):
    """
    Load the training data.

    Parameters
    ----------
    path : str, optional
        Base path to the data directory. Default is the current directory.

    Returns
    -------
    X : numpy.ndarray
        Array of text embeddings.
    y : numpy.ndarray
        Array of images.
    """
    return _get_data(path, split="train")


def get_test_data(path="."):
    """
    Load the test data.

    Parameters
    ----------
    path : str, optional
        Base path to the data directory. Default is the current directory.

    Returns
    -------
    X : numpy.ndarray
        Array of text embeddings.
    y : numpy.ndarray
        Array of images.
    """
    return _get_data(path, split="test")


def get_cv(X, y, random_state=42):
    """
    Get cross-validation splits using KFold.

    This function defines a cross-validation scheme to split the training data.
    Here we use KFold with shuffling to create 5 folds, setting a random seed
    for reproducibility.

    Parameters
    ----------
    X : array-like
        Feature array.
    y : array-like
        Target array.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    generator
        A generator yielding (train_idx, test_idx) for each fold.
    """
    kf = KFold(n_splits=2, shuffle=True, random_state=random_state)
    return kf.split(X)
