"""
This provides the base class of any Gaussian factor model, such as 
(probabilistic) PCA, supervised PCA, and factor analysis, as well as 
supervised and adversarial derivatives. 

Implementing an extension model requires that the following methods be 
implemented
    fit - Learns the model given data (and optionally supervised/adversarial
                       labels
    get_covariance - Given a fitted model returns the covariance matrix

For the remaining shared methods computing likelihoods etc, there are two 
options, compute the covariance matrix then use the default methods from
the covariance module, or use the Sherman-Woodbury matrix identity to 
invert the matrix more efficiently. Currently only the first is implemented
but left the options for future implementation.

Objects
-------
BaseGaussianFactorModel(_BaseSGDModel)
    Base class of all factor analysis models implementing any shared 
    Gaussian methods.

BaseSGDModel(BaseGaussianFactorModel)
    This is the base class of models that use Tensorflow stochastic
    gradient descent for inference. This reduces boilerplate code 
    associated with some of the standard methods etc.

Methods
-------
None
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from numpy import linalg as la
from numpy.typing import NDArray
from torch import Tensor

from ._base_covariance import (
    _conditional_score,
    _conditional_score_samples,
    _conditional_score_samples_sherman_woodbury,
    _conditional_score_sherman_woodbury,
    _entropy,
    _entropy_subset,
    _get_stable_rank,
    _marginal_score,
    _marginal_score_samples,
    _marginal_score_samples_sherman_woodbury,
    _marginal_score_sherman_woodbury,
    _mutual_information,
    _score,
    _score_samples,
    _score_samples_sherman_woodbury,
    _score_sherman_woodbury,
    inv_sherman_woodbury_fa,
)


def _get_projection_matrix(W_: Tensor, sigma_: Tensor, device: Any):
    """
    Compute the posterior parameters of the latent distribution p(S|X).

    Currently implemented only for PPCA due to its closed-form expression.
    Computes the projection matrix and posterior covariance via the matrix
    M = W W^T + sigma * I.

    Parameters
    ----------
    W_ : Tensor of shape (n_components, p)
        The factor loadings matrix.

    sigma_ : Tensor
        Scalar isotropic noise variance.

    device : Any
        The PyTorch device (CPU or CUDA) on which to perform computations.

    Returns
    -------
    Proj_X : Tensor of shape (n_components, p)
        Projection matrix such that ``Proj_X @ X`` gives ``E[S|X]``.

    Cov : Tensor of shape (n_components, n_components)
        Posterior covariance ``Var(S|X)``.
    """
    n_components = int(W_.shape[0])
    eye = torch.tensor(np.eye(n_components).astype(np.float32), device=device)
    M_init = torch.matmul(W_, torch.transpose(W_, 0, 1))
    M_end = sigma_ * eye
    M = M_init + M_end
    Proj_X = torch.linalg.solve(M, W_)
    Cov = torch.linalg.inv(M) * sigma_
    return Proj_X, Cov


def kl_divergence_vae(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the KL divergence from N(mu1, diag(sigma1)) to N(0, I).

    Computes ``KL(N(mu1, sigma1) || N(0, I))`` using the closed-form
    expression for diagonal Gaussian distributions::

        KL = 0.5 * (tr(sigma1) + mu1^T mu1 - log(det(sigma1)) - d)

    where ``d`` is the dimensionality of the latent space.

    Parameters
    ----------
    mu1 : torch.Tensor of shape (batch_size, d)
        Mean vector of the approximate posterior distribution.

    sigma1 : torch.Tensor of shape (batch_size, d)
        Diagonal of the covariance matrix of the approximate posterior.
        Expected to contain positive values.

    Returns
    -------
    kl_div : torch.Tensor of shape (batch_size,)
        Per-sample KL divergence values.
    """
    d = sigma1.shape[1]  # Dimensionality of the latent space
    term1 = -torch.sum(
        torch.log(sigma1), dim=1
    )  # Log determinant of diagonal covariance
    term2 = -d  # Negative of the latent space dimensionality
    term3 = torch.sum(
        sigma1, dim=1
    )  # Trace of diagonal covariance (sum of diagonal)
    term4 = torch.sum(torch.square(mu1), dim=1)  # Sum of squares
    kl_div = 0.5 * (term1 + term2 + term3 + term4)
    return kl_div


class BaseGaussianFactorModel(ABC):
    def __init__(self, n_components=2):
        """
        Base class for Gaussian factor models. Not intended to be instantiated directly.

        Parameters
        ----------
        n_components : int, default=2
            The dimensionality of the latent space.
        """
        self.n_components = int(n_components)
        self.W_ = None

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit the model to data.

        Subclasses must implement this method. Supervised subclasses
        additionally accept a label array ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data matrix.

        *args : additional positional arguments
            Passed to the subclass implementation.

        **kwargs : additional keyword arguments
            Passed to the subclass implementation.

        Returns
        -------
        self : object
            Fitted estimator.
        """

    @abstractmethod
    def get_covariance(self) -> NDArray[np.float64]:
        """
        Compute the covariance matrix implied by the fitted model parameters.

        Returns
        -------
        covariance : ndarray of shape (n_features, n_features)
            The model covariance matrix.
        """

    @abstractmethod
    def get_noise(self) -> NDArray[np.float64]:
        """
        Return the observational noise as a diagonal matrix.

        Returns
        -------
        Lambda : ndarray of shape (n_features, n_features)
            Diagonal noise matrix.
        """

    def get_precision(
        self, sherman_woodbury: bool = False
    ) -> NDArray[np.float64]:
        """
        Compute the precision matrix (inverse of the covariance matrix).

        Parameters
        ----------
        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury matrix identity for
            efficient inversion, exploiting the low-rank-plus-diagonal
            structure of the covariance. Requires ``self.get_noise()`` to
            return a diagonal matrix.

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            The precision matrix, i.e. the inverse of the covariance matrix.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return la.inv(covariance)

        return inv_sherman_woodbury_fa(self.get_noise(), self.W_)

    def get_stable_rank(self) -> np.float64:
        """
        Compute the stable rank of the covariance matrix.

        The stable rank is defined as ``||A||_F^2 / ||A||_2^2``, i.e. the
        ratio of the squared Frobenius norm to the squared spectral norm.
        This provides a statistically robust approximation to the matrix rank.
        See Vershynin, *High-Dimensional Probability* for a detailed discussion.

        Returns
        -------
        srank : np.float64
            The stable rank of the covariance matrix.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        return _get_stable_rank(self.get_covariance())

    def transform(self, X: NDArray[np.float64], sherman_woodbury: bool = False):
        """
        Project data into the latent factor space.

        Computes posterior mean estimates of the latent variables given
        the observations ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data matrix to transform.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity to exploit the
            low-rank-plus-diagonal structure for efficient computation.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
            Estimated latent factor scores.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            prec = self.get_precision()
            coefs = np.dot(self.W_, prec)
        else:
            W_ = self.W_
            M = np.dot(W_, W_.T) + self.sigma2_ * np.eye(self.n_components)
            Mi = la.inv(M)
            coefs = np.dot(Mi, W_)
            # raise NotImplementedError("Subclass PCA required for this Sherman Woodbury")

        return np.dot(X, coefs.T)

    def transform_subset(
        self,
        X: NDArray[np.float64],
        observed_feature_idxs: NDArray[np.float64],
        sherman_woodbury: bool = False,
    ) -> NDArray[np.float64]:
        """
        Project partially-observed data into the latent factor space.

        Computes latent variable estimates using only the observed features
        indicated by ``observed_feature_idxs``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_observed)
            Data matrix containing only the observed features, where
            ``n_observed = observed_feature_idxs.sum()``.

        observed_feature_idxs : ndarray of shape (n_features,)
            Boolean or integer index array indicating which features are
            observed (value 1) versus missing (value 0).

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity for efficient
            computation under low-rank-plus-diagonal covariance structure.

        Returns
        -------
        S : ndarray of shape (n_samples, n_components)
            Estimated latent factor scores.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        Wo = self.W_[:, observed_feature_idxs]
        if sherman_woodbury is False:
            covariance = self.get_covariance()
            cov_sub = covariance[
                np.ix_(observed_feature_idxs, observed_feature_idxs)
            ]
            coefs = np.dot(Wo, la.inv(cov_sub))
        else:
            M = np.dot(Wo, Wo.T) + self.sigma2_ * np.eye(self.n_components)
            Mi = la.inv(M)
            coefs = np.dot(Mi, Wo)
        return np.dot(X, coefs.T)

    def conditional_score(
        self,
        X: NDArray[np.float64],
        observed_feature_idxs: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
        sherman_woodbury: bool = False,
    ) -> np.float64:
        """
        Compute the weighted average conditional log-likelihood.

        Evaluates ``mean(log p(X[idx==0] | X[idx==1], covariance))``
        over all samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix (all features, both observed and
            unobserved).

        observed_feature_idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 indicates an observed
            feature and 0 indicates a missing feature.

        weights : ndarray of shape (n_samples,), default=None
            Optional sample weights. If ``None``, all samples are weighted
            equally. Weights are normalized to have mean 1.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity for efficient
            computation under low-rank-plus-diagonal covariance structure.

        Returns
        -------
        avg_score : np.float64
            Weighted average conditional log-likelihood.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _conditional_score(
                covariance, X, observed_feature_idxs, weights=weights
            )

        return _conditional_score_sherman_woodbury(
            self.get_noise(),
            self.W_,
            X,
            observed_feature_idxs,
            weights=weights,
        )

    def conditional_score_samples(
        self,
        X: NDArray[np.float64],
        observed_feature_idxs: NDArray[np.float64],
        sherman_woodbury: bool = False,
    ) -> NDArray[np.float64]:
        """
        Compute the per-sample conditional log-likelihood.

        Evaluates ``log p(X[idx==0] | X[idx==1], covariance)`` for each
        sample individually.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix (all features).

        observed_feature_idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 indicates an observed
            feature and 0 indicates a missing feature.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury matrix identity for
            efficient computation under low-rank-plus-diagonal covariance.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Per-sample conditional log-likelihood values.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _conditional_score_samples(
                covariance, X, observed_feature_idxs
            )

        return _conditional_score_samples_sherman_woodbury(
            self.get_noise(), self.W_, X, observed_feature_idxs
        )

    def marginal_score(
        self,
        X: NDArray[np.float64],
        observed_feature_idxs: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
        sherman_woodbury: bool = False,
    ) -> np.float64:
        """
        Compute the weighted average marginal log-likelihood of a feature subset.

        Evaluates ``mean(log p(X[:, idx==1], covariance_sub))`` over all
        samples, where the marginal distribution is derived from the model
        covariance restricted to the observed features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_observed)
            Data for the observed features only, where
            ``n_observed = observed_feature_idxs.sum()``.

        observed_feature_idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 indicates an observed
            feature.

        weights : ndarray of shape (n_samples,), default=None
            Optional sample weights. If ``None``, all samples are weighted
            equally.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity for efficient
            computation under low-rank-plus-diagonal covariance structure.

        Returns
        -------
        avg_score : np.float64
            Weighted average marginal log-likelihood.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _marginal_score(
                covariance, X, observed_feature_idxs, weights=weights
            )

        return _marginal_score_sherman_woodbury(
            self.get_noise(),
            self.W_,
            X,
            observed_feature_idxs,
            weights=weights,
        )

    def marginal_score_samples(
        self,
        X: NDArray,
        observed_feature_idxs: NDArray,
        sherman_woodbury: bool = False,
    ):
        """
        Compute the per-sample marginal log-likelihood of a feature subset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_observed)
            Data for the observed features only, where
            ``n_observed = observed_feature_idxs.sum()``.

        observed_feature_idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 indicates an observed
            feature.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity for efficient
            computation under low-rank-plus-diagonal covariance structure.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Per-sample marginal log-likelihood values.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            covariance = self.get_covariance()
            return _marginal_score_samples(covariance, X, observed_feature_idxs)

        return _marginal_score_samples_sherman_woodbury(
            self.get_noise(), self.W_, X, observed_feature_idxs
        )

    def score(
        self,
        X: NDArray[np.float64],
        weights: NDArray[np.float64] | None = None,
        sherman_woodbury: bool = False,
    ) -> np.float64:
        """
        Compute the weighted average log-likelihood of the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        weights : ndarray of shape (n_samples,), default=None
            Optional sample weights. If ``None``, all samples are weighted
            equally. Weights are normalized to have mean 1.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity for efficient
            computation under low-rank-plus-diagonal covariance structure.

        Returns
        -------
        avg_score : np.float64
            Weighted average log-likelihood.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            return _score(self.get_covariance(), X, weights=weights)

        return _score_sherman_woodbury(
            self.get_noise(), self.W_, X, weights=weights
        )

    def score_samples(
        self, X: NDArray[np.float64], sherman_woodbury: bool = False
    ) -> NDArray[np.float64]:
        """
        Compute the per-sample log-likelihood.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity for efficient
            computation under low-rank-plus-diagonal covariance structure.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Per-sample log-likelihood values.
        """
        if self.W_ is None:
            raise ValueError("Model has not been fit yet")

        if sherman_woodbury is False:
            return _score_samples(self.get_covariance(), X)

        return _score_samples_sherman_woodbury(self.get_noise(), self.W_, X)

    def get_entropy(self) -> np.float64:
        """
        Compute the differential entropy of the model's Gaussian distribution.

        Returns
        -------
        entropy : np.float64
            Differential entropy of the distribution in nats.
        """
        return _entropy(self.get_covariance())

    def get_entropy_subset(
        self, observed_feature_idxs: NDArray[np.float64]
    ) -> np.float64:
        """
        Compute the differential entropy of a marginal feature subset.

        Parameters
        ----------
        observed_feature_idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 indicates the features
            to include in the entropy computation.

        Returns
        -------
        entropy : np.float64
            Differential entropy of the marginal distribution in nats.
        """
        return _entropy_subset(self.get_covariance(), observed_feature_idxs)

    def mutual_information(
        self,
        observed_feature_idxs1: NDArray[np.float64],
        observed_feature_idxs2: NDArray[np.float64],
    ) -> np.float64:
        """
        Compute the mutual information between two groups of features.

        Uses the fitted model covariance to compute the mutual information
        ``I(X1 ; X2)`` based on the Gaussian entropy formula.

        Parameters
        ----------
        observed_feature_idxs1 : ndarray of shape (n_features,)
            Boolean or integer index array selecting the first group of
            features.

        observed_feature_idxs2 : ndarray of shape (n_features,)
            Boolean or integer index array selecting the second group of
            features.

        Returns
        -------
        mutual_information : np.float64
            Mutual information between the two feature groups in nats.
        """
        covariance = self.get_covariance()
        return _mutual_information(
            covariance, observed_feature_idxs1, observed_feature_idxs2
        )


class BasePCASGDModel(BaseGaussianFactorModel):
    def __init__(
        self,
        n_components: int = 2,
        training_options: dict[str, Any] | None = None,
        prior_options: dict[str, Any] | None = None,
    ) -> None:
        """
        Base class for factor models fitted via stochastic gradient descent.

        Provides shared infrastructure (loss tracking, device handling,
        optimizer setup) for SGD-based Gaussian factor models.

        Parameters
        ----------
        n_components : int, default=2
            The dimensionality of the latent space.

        training_options : dict, default=None
            Options for stochastic gradient descent. Keys and defaults are
            set by ``_fill_training_options``.

        prior_options : dict, default=None
            Options for the prior on model parameters. Keys and defaults
            are set by ``_fill_prior_options`` in the subclass.
        """
        super().__init__(n_components=n_components)

        if training_options is None:
            training_options = {}
        if prior_options is None:
            prior_options = {}

        self.training_options = self._fill_training_options(training_options)
        self.prior_options = self._fill_prior_options(prior_options)

    def _fill_training_options(
        self, training_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fill in default training options for stochastic gradient descent.

        Merges user-supplied options with defaults. Raises ``ValueError``
        if any unrecognized keys are present.

        Parameters
        ----------
        training_options : dict
            User-supplied training options. Recognized keys:

            - ``n_iterations`` : int, default=3000
                Number of SGD iterations.
            - ``learning_rate`` : float, default=1e-2
                Step size for the optimizer.
            - ``use_gpu`` : bool, default=True
                Whether to use a GPU if available.
            - ``method`` : str, default='Nadam'
                Optimization algorithm name.
            - ``batch_size`` : int, default=100
                Mini-batch size per iteration.
            - ``momentum`` : float, default=0.9
                Momentum coefficient for the SGD optimizer.

        Returns
        -------
        tops : dict
            Merged dictionary of training options with defaults applied.
        """
        default_options = {
            "n_iterations": 3000,
            "learning_rate": 1e-2,
            "use_gpu": True,
            "method": "Nadam",
            "batch_size": 100,
            "momentum": 0.9,
        }

        unexpected_but_present_keys = set(training_options.keys()) - set(
            default_options.keys()
        )
        if unexpected_but_present_keys:
            raise ValueError(
                "the following training options were unrecognized but provided..."
            )

        tops = {**default_options, **training_options}
        return tops

    def _initialize_save_losses(self) -> None:
        """
        Initialize arrays to record training losses at each iteration.

        Sets the following attributes:

        Attributes
        ----------
        losses_likelihood : ndarray of shape (n_iterations,)
            Per-iteration log-likelihood values.

        losses_prior : ndarray of shape (n_iterations,)
            Per-iteration log-prior values.

        losses_posterior : ndarray of shape (n_iterations,)
            Per-iteration log-posterior values.
        """
        n_iterations = self.training_options["n_iterations"]
        self.losses_likelihood = np.empty(n_iterations)
        self.losses_prior = np.empty(n_iterations)
        self.losses_posterior = np.empty(n_iterations)

    def _save_losses(
        self,
        i: int,
        device: Any,
        log_likelihood: Tensor,
        log_prior: Tensor | NDArray[np.float64],
        log_posterior: Tensor,
    ) -> None:
        """
        Record training loss values for the current iteration.

        Parameters
        ----------
        i : int
            Current training iteration index.

        device : Any
            PyTorch device (CPU or CUDA) used during training.

        log_likelihood : Tensor
            Log-likelihood value at iteration ``i``.

        log_prior : Tensor or ndarray of shape ()
            Log-prior value at iteration ``i``.

        log_posterior : Tensor
            Log-posterior value at iteration ``i``.
        """
        if device.type == "cuda":
            self.losses_likelihood[i] = log_likelihood.detach().cpu().numpy()
            if isinstance(log_prior, np.ndarray):
                self.losses_prior[i] = log_prior
            else:
                self.losses_prior[i] = log_prior.detach().cpu().numpy()
            self.losses_posterior[i] = log_posterior.detach().cpu().numpy()
        else:
            self.losses_likelihood[i] = log_likelihood.detach().numpy()
            if isinstance(log_prior, np.ndarray):
                self.losses_prior[i] = log_prior
            else:
                self.losses_prior[i] = log_prior.detach().numpy()
            self.losses_posterior[i] = log_posterior.detach().numpy()

    def _transform_training_data(
        self, device: Any, *args: NDArray
    ) -> list[Tensor]:
        """
        Convert numpy arrays to float32 PyTorch tensors on the target device.

        Parameters
        ----------
        device : Any
            PyTorch device (CPU or CUDA) to place the tensors on.

        *args : ndarray
            One or more numpy arrays to convert.

        Returns
        -------
        out : list of Tensor
            Tensors corresponding to each input array, cast to float32.
        """
        out = []
        for arg in args:
            out.append(torch.tensor(arg.astype(np.float32)).to(device))
        return out

    @abstractmethod
    def _create_prior(self, device: Any):
        """
        Construct the log-prior function for the model parameters.

        Parameters
        ----------
        device : Any
            PyTorch device (CPU or CUDA) on which to place prior tensors.

        Returns
        -------
        log_prior : callable
            A function that accepts the list of trainable variable tensors
            and returns a scalar Tensor representing the log-prior density.
        """
