import numpy as np
from scipy.linalg import sqrtm
from .base import BaseScoreType

class FID(BaseScoreType):
    """
    Fréchet Inception Distance (FID) for image generation.
    
    This metric computes the difference between the statistics of the
    generated images and the real images. Lower values indicate that the
    generated images are closer to the real ones.
    
    Note: This implementation flattens images and computes covariances
    directly. For more robust evaluations, it is common to extract features
    using a pre-trained network such as Inception.
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')
    
    def __init__(self, name='FID', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        """
        Compute the FID between ground truth images and generated images.
        
        Parameters
        ----------
        y_true : numpy array
            Ground truth images. Shape: (n_samples, height, width, channels).
        y_pred : numpy array
            Generated images. Shape: (n_samples, height, width, channels).
        
        Returns
        -------
        float
            The computed FID score.
        """
        # Flatten images to vectors
        y_true_flat = y_true.reshape(y_true.shape[0], -1)
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        
        # Compute mean vectors
        mu_true = np.mean(y_true_flat, axis=0)
        mu_pred = np.mean(y_pred_flat, axis=0)
        
        # Compute covariance matrices
        sigma_true = np.cov(y_true_flat, rowvar=False)
        sigma_pred = np.cov(y_pred_flat, rowvar=False)
        
        # Compute squared difference between means
        diff = mu_true - mu_pred
        diff_squared = np.sum(diff**2)
        
        # Compute the square root of the product of covariance matrices
        covmean = sqrtm(sigma_true.dot(sigma_pred))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff_squared + np.trace(sigma_true + sigma_pred - 2 * covmean)
        return fid
