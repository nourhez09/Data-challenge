import os
import numpy as np
from rampwf.utils.importing import import_module_from_source
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.utils.data import Dataset

class ConditionedImageGenerator(object):
    """
    ConditionedImageGenerator workflow.

    This workflow is designed for text-conditioned image generation tasks.
    Submissions need to contain one file, 'generator.py', which defines a function
    'get_generator()'. The function should return a model that implements both
    fit() and predict() methods.

    The fit() method should accept training text embeddings (conditions) and
    corresponding ground truth images. The predict() method should take text
    embeddings as input and output generated images.
    """
    
    # Define the required submission file(s)
    workflow_element_names = ['CondImageGenerator']

    def __init__(self):
        pass

    def train_submission(self, module_path, X_train, y_train, train_is=None):
        """
        Train the text-conditioned image generator.

        Parameters
        ----------
        module_path : str
            Path to the submission module directory, which must contain generator.py.
        X_train : numpy array
            Array of training text embeddings, expected shape: (n_samples, 768).
        y_train : numpy array
            Array of training images, expected shape: (n_samples, height, width, channels).
        train_is : array-like, optional
            Indices for training; if provided, the training data will be sliced.

        Returns
        -------
        model
            The trained generator model.
        """
        if train_is is not None:
            X_train = X_train[train_is]
            y_train = y_train[train_is]

        # Import the submission's generator.py file.
        generator_module = import_module_from_source(
            os.path.join(module_path, self.workflow_element_names[0] + '.py'),
            self.workflow_element_names[0],
            sanitize=True
        )
        # Expect the submission to have a function named get_generator()
        model = generator_module.ConditionalVAE()
        # Train the generator using the training data (text embeddings and images)
        #convert X_train, y_train to torch tensors
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        #create dataloader
        dataset = TensorDataset(X_train, y_train)
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.fit(dataloader)
        return model

    def test_submission(self, trained_model, X_test, y_test=None):
        """
        Generate images from test text embeddings.

        Parameters
        ----------
        trained_model : object
            The generator model returned by train_submission.
        X_test : numpy array
            Array of test text embeddings, expected shape: (n_samples, 768).
        y_test : numpy array
            Array of test images, expected shape: (n_samples, height, width, channels).

        Returns
        -------
        numpy array
            Array of generated images, expected shape: (n_samples, height, width, channels).
        """
        #convert X_test, y_test to torch tensors
        X_test = torch.tensor(X_test)
        if y_test is None:
            y_test = torch.zeros((X_test.shape[0], 128, 128, 3))
        else:
            y_test = torch.tensor(y_test)
        #create dataloader
        dataset = TensorDataset(X_test, y_test)
        batch_size = 64
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        y_pred = trained_model.predict(dataloader)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        return y_pred
