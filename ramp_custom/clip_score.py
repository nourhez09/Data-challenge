import numpy as np
from numpy.linalg import norm
from rampwf.score_types.base import BaseScoreType

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18

class CLIPScore(BaseScoreType):
    """
    CLIP-like Score for text-conditioned image generation.

    This metric:
      1. Accepts pre-computed text embeddings (shape: (n_samples, 768)).
      2. Computes image embeddings from generated images (numpy array of shape (n_samples, H, W, C))
         using a ResNet-18 backbone.
      3. Projects the ResNet features (512-dim) to 768 dimensions so they match the text embeddings.
      4. Normalizes both embeddings and computes the cosine similarity.
      5. Returns the inverse of the mean cosine similarity (so that lower scores are better).
    """
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')
    
    def __init__(self, name='clip score', precision=2):
        self.name = name
        self.precision = precision

        # 1) Initialize a ResNet-18 backbone and remove its final classification layer.
        self.image_model = resnet18(pretrained=True)
        self.image_model.fc = nn.Identity()  # Now outputs a 512-dim feature vector

        # 2) Define a linear projection from 512 -> 768 to match text embeddings.
        self.projection = nn.Linear(512, 768, bias=False)

        # Set models to evaluation mode and freeze parameters.
        self.image_model.eval()
        for param in self.image_model.parameters():
            param.requires_grad = False
        for param in self.projection.parameters():
            param.requires_grad = False

        # # 3) Define a transform that normalizes images as expected by ResNet.
        # self.transform = T.Compose([
        #     T.ConvertImageDtype(torch.float),  # Convert image to float (0,1)
        #     T.Normalize(mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225]),
        # ])

    def __call__(self, text_embeddings: np.ndarray, generated_images: np.ndarray) -> float:
        """
        Compute a CLIP-like score between text embeddings and generated images.

        Parameters
        ----------
        text_embeddings : np.ndarray
            Pre-computed text embeddings. Expected shape: (n_samples, 768).
        generated_images : np.ndarray
            Generated images. Expected shape: (n_samples, height, width, channels) with 3 channels (RGB).

        Returns
        -------
        float
            The inverse of the mean cosine similarity between text and image embeddings.
        """

        # -------------------------------------------------------
        # 1) Preprocess images and convert to Torch tensors.
        # -------------------------------------------------------
        images_tensor = torch.from_numpy(generated_images)
        images_tensor = images_tensor.float() / 255.0  # Scale pixel values from [0,255] to [0,1]
        #images_tensor = self.transform(images_tensor)

        # -------------------------------------------------------
        # 2) Extract features with the ResNet-18 backbone.
        # -------------------------------------------------------
        with torch.no_grad():
            features = self.image_model(images_tensor)  # (n_samples, 512)

        # -------------------------------------------------------
        # 3) Project features to match the text embedding dimension (768).
        # -------------------------------------------------------
        with torch.no_grad():
            image_embeddings_torch = self.projection(features)  # (n_samples, 768)

        # Convert the image embeddings to a NumPy array.
        image_embeddings = image_embeddings_torch.cpu().numpy()

        # -------------------------------------------------------
        # 4) Normalize embeddings and compute cosine similarity.
        # -------------------------------------------------------
        text_norm = text_embeddings / (norm(text_embeddings, axis=1, keepdims=True) + 1e-8)
        if isinstance(text_norm, torch.Tensor):
            text_norm = text_norm.numpy()
        image_norm = image_embeddings / (norm(image_embeddings, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.abs(np.sum(text_norm * image_norm, axis=1))  # Cosine similarity for each sample
        mean_cos_sim = np.mean(cos_sim)

        # -------------------------------------------------------
        # 5) Return the inverse of the mean similarity (lower is better).
        # -------------------------------------------------------
        clip_score = 1.0 / (mean_cos_sim + 1e-8)
        return clip_score
    
    def score_function(self, ground_truths, predictions) -> float:
        X_test =  predictions.text_embeddings
        y_test = predictions.y_pred
        return self.__call__(X_test, y_test)