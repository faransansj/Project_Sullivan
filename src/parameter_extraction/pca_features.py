"""
PCA-based Feature Extraction for Articulatory Parameters

This module uses Principal Component Analysis to extract low-dimensional
representations from vocal tract segmentation masks.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Optional
import pickle


class PCAFeatureExtractor:
    """
    Extract PCA-based articulatory features from segmentation masks.

    This extractor flattens segmentation masks and applies PCA to reduce
    dimensionality. It can be trained on a dataset and then used to transform
    new samples.

    Parameters
    ----------
    n_components : int, default=10
        Number of PCA components to extract
    whiten : bool, default=False
        Whether to whiten the components (normalize variance)
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_components: int = 10,
        whiten: bool = False,
        random_state: Optional[int] = 42
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

        self.pca = PCA(
            n_components=n_components,
            whiten=whiten,
            random_state=random_state
        )
        self._is_fitted = False
        self._input_shape = None

    def fit(self, segmentations: np.ndarray) -> 'PCAFeatureExtractor':
        """
        Fit PCA model on a batch of segmentation masks.

        Parameters
        ----------
        segmentations : np.ndarray
            Batch of segmentation masks, shape (num_samples, height, width)

        Returns
        -------
        self : PCAFeatureExtractor
            Fitted extractor
        """
        num_samples, height, width = segmentations.shape
        self._input_shape = (height, width)

        # Flatten masks to (num_samples, height*width)
        flattened = segmentations.reshape(num_samples, -1).astype(np.float32)

        # Fit PCA
        self.pca.fit(flattened)
        self._is_fitted = True

        return self

    def transform(self, segmentations: np.ndarray) -> np.ndarray:
        """
        Transform segmentation masks to PCA features.

        Parameters
        ----------
        segmentations : np.ndarray
            Batch of segmentation masks, shape (num_samples, height, width)

        Returns
        -------
        features : np.ndarray
            PCA features, shape (num_samples, n_components)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA extractor must be fitted before transform. Call fit() first.")

        num_samples, height, width = segmentations.shape

        if (height, width) != self._input_shape:
            raise ValueError(
                f"Input shape {(height, width)} doesn't match fitted shape {self._input_shape}"
            )

        # Flatten and transform
        flattened = segmentations.reshape(num_samples, -1).astype(np.float32)
        features = self.pca.transform(flattened)

        return features.astype(np.float32)

    def fit_transform(self, segmentations: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform in one step.

        Parameters
        ----------
        segmentations : np.ndarray
            Batch of segmentation masks, shape (num_samples, height, width)

        Returns
        -------
        features : np.ndarray
            PCA features, shape (num_samples, n_components)
        """
        self.fit(segmentations)
        return self.transform(segmentations)

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Reconstruct segmentation masks from PCA features.

        Parameters
        ----------
        features : np.ndarray
            PCA features, shape (num_samples, n_components)

        Returns
        -------
        segmentations : np.ndarray
            Reconstructed masks, shape (num_samples, height, width)
        """
        if not self._is_fitted:
            raise RuntimeError("PCA extractor must be fitted before inverse_transform.")

        # Inverse transform to flattened
        flattened = self.pca.inverse_transform(features)

        # Reshape to original shape
        height, width = self._input_shape
        segmentations = flattened.reshape(-1, height, width)

        # Clip to valid range [0, 3] for class indices
        segmentations = np.clip(segmentations, 0, 3)

        return segmentations.astype(np.uint8)

    def save(self, filepath: str):
        """
        Save fitted PCA model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model (pickle format)
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted PCA extractor.")

        model_data = {
            'pca': self.pca,
            'n_components': self.n_components,
            'whiten': self.whiten,
            'input_shape': self._input_shape,
            'is_fitted': self._is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    @classmethod
    def load(cls, filepath: str) -> 'PCAFeatureExtractor':
        """
        Load fitted PCA model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved model

        Returns
        -------
        extractor : PCAFeatureExtractor
            Loaded and fitted extractor
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        extractor = cls(
            n_components=model_data['n_components'],
            whiten=model_data['whiten']
        )
        extractor.pca = model_data['pca']
        extractor._input_shape = model_data['input_shape']
        extractor._is_fitted = model_data['is_fitted']

        return extractor

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio of each component."""
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted yet.")
        return self.pca.explained_variance_ratio_

    @property
    def total_explained_variance(self) -> float:
        """Get total explained variance by all components."""
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted yet.")
        return np.sum(self.pca.explained_variance_ratio_)

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (f"PCAFeatureExtractor(n_components={self.n_components}, "
                f"whiten={self.whiten}, {fitted_str})")
