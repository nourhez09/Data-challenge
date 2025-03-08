"""
Predictions for text-conditioned image generation.

y_pred should be a 4D numpy array with shape:
    (n_samples, height, width, channels)
where each element corresponds to a generated image.
"""
import warnings
import numpy as np
from ramp_custom.FID import FID
from rampwf.prediction_types.base import BasePrediction

class Generation(BasePrediction):
    """
    Custom prediction class for text-conditioned image generation.
    
    This class handles predictions as numpy arrays representing images.
    It verifies that the images have the expected dimensions and provides
    functionality to combine multiple predictions (e.g., from cross-validation or ensembling).
    """
    def __init__(self, y_pred=None, y_true=None, n_samples=None, img_shape=(3, 128, 128), fold_is=None):
        """
        Initialize the Generation object.

        Parameters
        ----------
        y_pred : numpy array, optional
            Array of generated images. Expected shape: (n_samples, height, width, channels).
        y_true : numpy array, optional
            Ground truth images. Only used to infer n_samples (shape) if y_pred is None.
        n_samples : int, optional
            Number of samples to initialize an empty container if neither y_pred nor y_true is provided.
        img_shape : tuple, default (64, 64, 3)
            Expected shape of each generated image (height, width, channels).
        """
        if y_pred is not None:
            if fold_is is not None:
                y_pred = y_pred[fold_is]
            self.y_pred = np.array(y_pred, dtype=np.float32)
        elif y_true is not None:
            # Initialize with zeros; shape inferred from y_true's first dimension.
            if fold_is is not None:
                y_true = y_true[fold_is]
            self.y_pred = np.array(y_true, dtype=np.float32)
        elif n_samples is not None:
            self.y_pred = np.empty((n_samples, *img_shape), dtype=np.float32)
        else:
            raise ValueError("Must provide y_pred, y_true, or n_samples for initialization.")
        
        self.img_shape = img_shape
        self.check_y_pred_dimensions()

    def check_y_pred_dimensions(self):
        """
        Ensure that y_pred has the correct dimensions:
        (n_samples, height, width, channels)
        """
        if isinstance(self.y_pred, FID):
            return
        if len(self.y_pred.shape) != 4:
            raise ValueError(f"y_pred must be 4D (n_samples, height, width, channels), got shape {self.y_pred.shape}")
        if self.y_pred.shape[1:] != self.img_shape:
            raise ValueError(f"Expected image shape {self.img_shape}, but got {self.y_pred.shape[1:]}")

    @property
    def valid_indexes(self):
        """
        Return valid indices for 4D image predictions.
        
        For image predictions, we consider an image valid if the first pixel of the first channel is not NaN.
        """
        return ~np.isnan(self.y_pred[:, 0, 0, 0])
    
    def set_valid_in_train(self, predictions, test_is):
        """
        Set a cross-validation slice for Generation predictions.

        Parameters
        ----------
        predictions : Generation
            A Generation instance with predictions to insert.
        test_is : array-like of booleans or indices
            The indices corresponding to the current test split.
        """
        # Ensure assignment is done on the full image tensor (all channels, height, width)
        self.y_pred[test_is, ...] = predictions.y_pred
    
    def set_slice(self, valid_indexes):
        """
        Extract a subset of predictions based on the given valid indexes.
        
        Parameters
        ----------
        valid_indexes : array-like
            Indices of the valid predictions to keep.
        """
        self.y_pred = self.y_pred[valid_indexes]


    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """
        Combine multiple Predictions instances by averaging their pixel values.
        
        This is useful when combining predictions from different CV folds or ensembling multiple models.

        Parameters
        ----------
        predictions_list : list of Predictions objects

        Returns
        -------
        Predictions
            A new Predictions instance containing the averaged images.
        """
        # Stack y_pred from each prediction instance (resulting shape: (n_models, n_samples, h, w, c))
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = np.array([pred.y_pred for pred in predictions_list])
        # Compute the mean across the first axis (i.e. across different predictions)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            y_comb = np.nanmean(y_comb_list, axis=0)
        return cls(y_pred=y_comb, img_shape=predictions_list[0].img_shape)

    def set_slice(self, valid_indexes):
        """
        Extract a subset of predictions based on the given valid indexes.
        
        Parameters
        ----------
        valid_indexes : array-like
            Indices of the valid predictions to keep.
        """
        self.y_pred = self.y_pred[valid_indexes]

    def __str__(self):
        return f"Predictions: {self.y_pred.shape[0]} samples, image shape {self.img_shape}"


def make_generation():
    """
    Factory function to create the generation prediction type.
    
    Returns
    -------
    Generation
        The custom prediction class for text-conditioned image generation.
    """
    return Generation
