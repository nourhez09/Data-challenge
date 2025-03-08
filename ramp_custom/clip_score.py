import numpy as np
from numpy.linalg import norm
from rampwf.score_types.base import BaseScoreType

class CLIPScore(BaseScoreType):
    """
    CLIP Score for text-conditioned image generation.
    
    This metric is based on the cosine similarity between the text embeddings
    (ground truth) and a dummy image embedding extracted from the generated images.
    A higher similarity indicates a better match; therefore, the inverse of the
    mean cosine similarity is returned so that lower scores are better.
    
    Note: In a real-world scenario, you would pass the generated images through a
    CLIP image encoder to obtain proper embeddings. Here, we provide a simple
    dummy projection for demonstration.
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')
    
    def __init__(self, name='clip score', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, text_embeddings, generated_images):
        """
        Compute the CLIP score between text embeddings and generated images.
        
        Parameters
        ----------
        text_embeddings : numpy array
            Ground truth text embeddings. Expected shape: (n_samples, 768).
        generated_images : numpy array
            Generated images. Expected shape: (n_samples, height, width, channels).
            For this dummy implementation, it is assumed that the images have 3 channels.
        
        Returns
        -------
        float
            The computed inverse CLIP score.
        """
        # Dummy image embedding: average over spatial dimensions → (n_samples, channels)
        img_avg = np.mean(generated_images, axis=(1, 2))
        
        # For 3-channel images, tile the average vector to 768 dimensions.
        if img_avg.shape[1] == 3:
            image_embeddings = np.tile(img_avg, (1, 256))
        else:
            # If channels are not 3, repeat values to match 768 dimensions.
            repeat_factor = int(768 / img_avg.shape[1])
            image_embeddings = np.repeat(img_avg, repeat_factor, axis=1)
        
        # Normalize both embeddings
        text_norm = text_embeddings / (norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
        image_norm = image_embeddings / (norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
        
        text_norm = text_norm.numpy()
        # Compute cosine similarity for each sample
        cos_sim = np.sum(text_norm * image_norm, axis=1)
        mean_cos_sim = np.mean(cos_sim)
        
        # Return the inverse of the mean cosine similarity so that lower scores are better.
        clip_score = 1.0 / (mean_cos_sim + 1e-8)
        return clip_score