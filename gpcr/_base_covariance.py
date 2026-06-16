"""
This module provides a range of functions and methods for advanced statistical
analysis and manipulation of covariance matrices. It includes tools for matrix
symmetrization, precision and stable rank calculations, along with predictive
modeling and entropy calculations using covariance matrices. The core
functionalities are encapsulated within the BaseCovariance object, enhancing
ease of use and integration in statistical and machine learning workflows.

Classes:
- BaseCovariance: Encapsulates covariance matrix operations.
  Methods:
  - __init__: Initialize with data validation.
  - get_precision: Retrieve precision matrices.
  - get_stable_rank: Calculate stable ranks.
  - predict: Predict missing data values.
  - conditional_score: Compute conditional scores for subsets.
  - conditional_score_samples: Score individual samples conditionally.
  - marginal_score: Calculate marginal scores for subsets.
  - marginal_score_samples: Score individual samples marginally.
  - score: General scoring function.
  - score_samples: Score individual samples.
  - entropy: Calculate entropy of the distribution.
  - entropy_subset: Calculate entropy for subsets.
  - mutual_information: Compute mutual information between sets.

Standalone Functions:
- _symmeterize_and_warning: Symmetrize covariance matrices with warnings.
- ldet_sherman_woodbury_fa: Calculate log determinant using factor analysis.
- ldet_sherman_woodbury_full: Calculate log determinant of a complex matrix.
- inv_sherman_woodbury_fa: Invert a matrix with diagonal and low-rank structure.
- inv_sherman_woodbury_full: Invert a complex matrix structured as A + UBV.
"""

from datetime import datetime as dt
from typing import Tuple

import numpy as np
import pytz
from numpy import linalg as la
from numpy.typing import NDArray


def _symmeterize_and_warning(cov: np.ndarray) -> np.ndarray:
    """
    Symmetrize the given square covariance matrix and warn if the symmetrization
    introduces significant changes.

    Parameters
    ----------
    cov : ndarray of shape (n_features, n_features)
        The covariance matrix to be symmetrized. It must be a square matrix.

    Returns
    -------
    cov_new : ndarray of shape (n_features, n_features)
        The symmetric matrix obtained by averaging the matrix with its transpose.

    Raises
    ------
    Warning
        If the relative mean squared error between the original and symmetrized
        covariance matrix exceeds 2%, indicating potential issues with the
        symmetrization process.
    """
    cov_new = (cov + cov.T) / 2
    diff_mse = np.mean((cov_new - cov) ** 2)
    cov_mse = np.mean(cov**2)
    if diff_mse / cov_mse > 0.02:
        print(
            "Warning: original matrix estimate deviated substantially from symmetric"
        )
    return cov_new


class BaseCovariance:
    def __init__(self) -> None:
        self.creationDate = dt.now(pytz.timezone("US/Pacific"))
        self.covariance: NDArray | None = None

    def get_precision(self) -> NDArray[np.float64]:
        """
        Compute the precision matrix (inverse of the covariance matrix).

        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            The precision matrix.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _get_precision(self.covariance)

    def get_stable_rank(self) -> np.float64:
        """
        Compute the stable rank of the covariance matrix.

        The stable rank is defined as ``||A||_F^2 / ||A||_2^2``.

        Returns
        -------
        srank : np.float64
            The stable rank of the covariance matrix.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """

        return _get_stable_rank(self.covariance)

    def predict(self, Xobs: NDArray, idxs: NDArray):
        """
        Predict missing features from observed features.

        Uses the conditional Gaussian formula to predict the unobserved
        features (where ``idxs == 0``) from the observed features
        (where ``idxs == 1``).

        Parameters
        ----------
        Xobs : ndarray of shape (n_samples, n_observed)
            Observed feature values, where ``n_observed = idxs.sum()``.

        idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 marks observed features
            and 0 marks features to predict.

        Returns
        -------
        preds : ndarray of shape (n_samples, n_missing)
            Predicted values for the unobserved features.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _predict(self.covariance, Xobs, idxs)

    def conditional_score(
        self, X: NDArray, idxs: NDArray, weights=None
    ) -> np.float64:
        """
        Compute the weighted average conditional log-likelihood.

        Evaluates ``mean(log p(X[idxs==0] | X[idxs==1], covariance))``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 marks observed features
            and 0 marks unobserved features.

        weights : ndarray of shape (n_samples,), default=None
            Optional sample weights. Normalized to have mean 1.

        Returns
        -------
        avg_score : np.float64
            Weighted average conditional log-likelihood.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _conditional_score(self.covariance, X, idxs, weights=weights)

    def conditional_score_samples(
        self, X: NDArray, idxs: NDArray
    ) -> NDArray[np.float64]:
        """
        Compute the per-sample conditional log-likelihood.

        Evaluates ``log p(X[idxs==0] | X[idxs==1], covariance)`` for each
        sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 marks observed features
            and 0 marks unobserved features.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Per-sample conditional log-likelihood values.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _conditional_score_samples(self.covariance, X, idxs)

    def marginal_score(
        self, X: NDArray, idxs: NDArray, weights=None
    ) -> np.float64:
        """
        Compute the weighted average marginal log-likelihood.

        Evaluates ``mean(log p(X[:, idxs==1], covariance_sub))``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_observed)
            Data for the observed features, where
            ``n_observed = idxs.sum()``.

        idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 marks the included
            features.

        weights : ndarray of shape (n_samples,), default=None
            Optional sample weights. Normalized to have mean 1.

        Returns
        -------
        avg_score : np.float64
            Weighted average marginal log-likelihood.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _marginal_score(self.covariance, X, idxs, weights=weights)

    def marginal_score_samples(
        self, X: NDArray, idxs: NDArray
    ) -> NDArray[np.float64]:
        """
        Compute the per-sample marginal log-likelihood.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_observed)
            Data for the observed features, where
            ``n_observed = idxs.sum()``.

        idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 marks the included
            features.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Per-sample marginal log-likelihood values.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _marginal_score_samples(self.covariance, X, idxs)

    def score(self, X: NDArray, weights=None) -> np.float64:
        """
        Compute the weighted average log-likelihood.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        weights : ndarray of shape (n_samples,), default=None
            Optional sample weights. Normalized to have mean 1.

        Returns
        -------
        avg_score : np.float64
            Weighted average log-likelihood.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _score(self.covariance, X, weights=weights)

    def score_samples(self, X: NDArray) -> NDArray[np.float64]:
        """
        Compute the per-sample log-likelihood.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Per-sample log-likelihood values.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _score_samples(self.covariance, X)

    def entropy(self) -> np.float64:
        """
        Compute the differential entropy of the fitted Gaussian distribution.

        Returns
        -------
        entropy : np.float64
            Differential entropy in nats.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _entropy(self.covariance)

    def entropy_subset(self, idxs: NDArray[np.float64]) -> np.float64:
        """
        Compute the differential entropy of a marginal feature subset.

        Parameters
        ----------
        idxs : ndarray of shape (n_features,)
            Boolean or integer index array where 1 marks the features to
            include in the entropy computation.

        Returns
        -------
        entropy : np.float64
            Differential entropy of the marginal distribution in nats.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _entropy_subset(self.covariance, idxs)

    def mutual_information(
        self, idxs1: NDArray[np.float64], idxs2: NDArray[np.float64]
    ) -> np.float64:
        """
        Compute the mutual information between two groups of features.

        Parameters
        ----------
        idxs1 : ndarray of shape (n_features,)
            Boolean or integer index array selecting the first group of
            features.

        idxs2 : ndarray of shape (n_features,)
            Boolean or integer index array selecting the second group of
            features.

        Returns
        -------
        mutual_information : np.float64
            Mutual information between the two feature groups in nats.

        Raises
        ------
        ValueError
            If the covariance matrix has not been set.
        """
        if self.covariance is None:
            raise ValueError("Covariance matrix has not been fit")

        return _mutual_information(self.covariance, idxs1, idxs2)

    def _test_inputs(self, X: NDArray[np.float64]) -> None:
        """
        Validate that ``X`` is a numpy array with no missing values.

        Parameters
        ----------
        X : ndarray
            Input data to validate.

        Raises
        ------
        ValueError
            If ``X`` is not a numpy array or contains NaN values.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is not a numpy array")
        if np.sum(np.isnan(X)) > 0:
            raise ValueError("Data has nans")


def _get_precision(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Gets the precision matrix defined as the inverse of the covariance

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    Returns
    -------
    precision : NDArray,(sum(p),sum(p))
        The inverse of the covariance matrix
    """
    return la.inv(covariance)


def _get_stable_rank(covariance: NDArray[np.float64]) -> np.float64:
    """
    Returns the stable rank defined as
    ||A||_F^2/||A||^2

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    Returns
    -------
    srank : np.float64
        The stable rank. See Vershynin High dimensional probability for
        discussion, but this is a statistically stable approximation to rank
    """
    singular_values = la.svd(covariance, compute_uv=False)
    return np.sum(singular_values**2) / singular_values[0] ** 2


def _predict(
    covariance: NDArray[np.float64],
    Xobs: NDArray[np.float64],
    idxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Predicts missing data using observed data. This uses the conditional
    Gaussian formula (see wikipedia multivariate gaussian). Solve allows
    us to avoid an explicit matrix inversion.

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    Xobs : NDArray,(N_samples,\\sum idxs)
        The observed data

    idxs: NDArray,(sum(p),)
        The observation locations

    Returns
    -------
    preds : NDArray,(N_samples,p-\\sum idxs)
        The predicted values
    """
    covariance_22 = covariance[np.ix_(idxs == 1, idxs == 1)]
    covariance_21 = covariance[np.ix_(idxs == 1, idxs == 0)]
    beta_bar = la.solve(covariance_22, covariance_21)

    return np.dot(Xobs[:, idxs == 1], beta_bar)


def _conditional_score(
    covariance: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> np.float64:
    """
    Returns the predictive log-likelihood of a subset of data.

    mean(log p(X[idx==1]|X[idx==0],covariance))

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    X : NDArray[np.float64],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float64],(sum(p),)
        The observation locations

    weights : NDArray[np.float64],(N,),default=None
        The optional weights on the samples. Don't have negative values.
        Average value forced to 1.

    Returns
    -------
    avg_score : np.float64
        Average log likelihood
    """
    weights = (
        np.ones(X.shape[0]) if weights is None else weights / np.mean(weights)
    )

    return np.mean(weights * _conditional_score_samples(covariance, X, idxs))


def _conditional_score_sherman_woodbury(
    Lambda: NDArray[np.float64],
    W: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> np.float64:
    """
    Returns the predictive log-likelihood of a subset of data.

    mean(log p(X[idx==1]|X[idx==0],covariance))

    Parameters
    ----------
    Lambda : NDArray[np.float64],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float64],(L,p)
        The low rank component

    X : NDArray[np.float64],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float64],(sum(p),)
        The observation locations

    weights : NDArray[np.float64] | None,(N,),default=None
        The optional weights on the samples. Don't have negative values.
        Average value forced to 1.

    Returns
    -------
    avg_score : np.float64
        Average log likelihood
    """
    weights = (
        np.ones(X.shape[0]) if weights is None else weights / np.mean(weights)
    )

    return np.mean(
        weights
        * _conditional_score_samples_sherman_woodbury(Lambda, W, X, idxs)
    )


def _conditional_score_samples(
    covariance: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Return the conditional log likelihood of each sample, that is

    log p(X[idx==1]|X[idx==0],covariance) = N(Sigma_10Sigma_00^{-1}x_0,
                                    Sigma_11-Sigma_10Sigma_00^{-1}Sigma_01)

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    X : NDArray[np.float64],(N,p)
        The centered data

    idxs: NDArray[np.float64],(p,)
        The observation locations

    Returns
    -------
    scores : np.float64
        Log likelihood for each sample
    """
    covariance_22 = covariance[np.ix_(idxs == 1, idxs == 1)]
    covariance_21 = covariance[np.ix_(idxs == 1, idxs == 0)]
    covariance_11 = covariance[np.ix_(idxs == 0, idxs == 0)]

    beta_bar = la.solve(covariance_22, covariance_21)
    Second_part = np.dot(covariance_21.T, beta_bar)

    covariance_bar = covariance_11 - Second_part

    mu_ = np.dot(X[:, idxs == 1], beta_bar)

    return _score_samples(covariance_bar, X[:, idxs == 0] - mu_)


def _conditional_score_samples_sherman_woodbury(
    Lambda: NDArray[np.float64],
    W: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Return the conditional log likelihood of each sample, that is

    log p(X[idx==1]|X[idx==0],covariance) = N(Sigma_10Sigma_00^{-1}x_0,
                                    Sigma_11-Sigma_10Sigma_00^{-1}Sigma_01)

    Parameters
    ----------
    Lambda : NDArray[np.float64],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float64],(L,p)
        The low rank component

    X : NDArray[np.float64],(N,p)
        The centered data

    idxs: NDArray[np.float64],(p,)
        The observation locations

    Returns
    -------
    scores : np.float64
        Log likelihood for each sample
    """
    I_L = np.eye(W.shape[0])
    X_obs = X[:, idxs == 1]
    X_miss = X[:, idxs == 0]
    W_obs = W[:, idxs == 1]
    W_miss = W[:, idxs == 0]
    Lo = Lambda[np.ix_(idxs == 1, idxs == 1)]
    Lm = Lambda[np.ix_(idxs == 0, idxs == 0)]

    C = np.dot(W_obs, la.inv(Lo))
    B = np.dot(C, W_obs.T)

    IpB = I_L + B
    back = la.solve(IpB, C)
    second = C - np.dot(B, back)
    coef = np.dot(W_miss.T, second)

    mu_ = np.dot(X_obs, coef.T)
    middle = B - np.dot(B, la.solve(IpB, B))
    term2 = np.dot(W_miss.T, np.dot(middle, W_miss))

    covariance11 = Lm + np.dot(W_miss.T, W_miss)
    covariance_bar = covariance11 - term2

    return _score_samples(covariance_bar, X_miss - mu_)


def _get_conditional_parameters(
    covariance: NDArray[np.float64], idxs: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes the distribution parameters p(X_miss|X_obs)

    Parameters
    ----------
    covariance : NDArray,(p,p)
        The covariance matrix

    idxs: NDArray,(p,)
        The observed covariates

    Returns
    -------
    beta_bar : array-like
        The predictive covariates

    covariance_bar : array-like
        Conditional covariance
    """
    covariance_sub = covariance[idxs == 1]
    covariance_22 = covariance_sub[:, idxs == 1]
    covariance_21 = covariance_sub[:, idxs == 0]
    covariance_nonsub = covariance[idxs == 0]
    covariance_11 = covariance_nonsub[:, idxs == 0]
    beta_bar = la.solve(covariance_22, covariance_21)
    Second_part = np.dot(covariance_21.T, beta_bar)

    covariance_bar = covariance_11 - Second_part

    return beta_bar.T, covariance_bar


def _get_conditional_parameters_sherman_woodbury(
    Lambda: NDArray[np.float64],
    W: NDArray[np.float64],
    idxs: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Computes the distribution parameters p(X_miss|X_obs)
    given that Sigma = WWT + Lambda

    Parameters
    ----------
    Lambda : NDArray,(p,p)
        The diagonal noise matrix

    W : NDArray,(L,p)
        The low rank component

    idxs: NDArray,(p,)
        The observed covariates

    Returns
    -------
    beta_bar : NDArray,(p,)
        The predictive covariates

    covariance_bar : NDArray,(p,p)
        Conditional covariance
    """
    I_L = np.eye(W.shape[0])
    W_obs = W[:, idxs == 1]
    W_miss = W[:, idxs == 0]
    Lo = Lambda[np.ix_(idxs == 1, idxs == 1)]
    Lm = Lambda[np.ix_(idxs == 0, idxs == 0)]

    C = np.dot(W_obs, la.inv(Lo))
    B = np.dot(C, W_obs.T)

    IpB = I_L + B
    back = la.solve(IpB, C)
    second = C - np.dot(B, back)
    coef = np.dot(W_miss.T, second)

    middle = B - np.dot(B, la.solve(IpB, B))
    term2 = np.dot(W_miss.T, np.dot(middle, W_miss))

    covariance11 = Lm + np.dot(W_miss.T, W_miss)
    covariance_bar = covariance11 - term2

    return coef, covariance_bar


def _marginal_score(
    covariance: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> np.float64:
    """
    Returns the marginal log-likelihood of a subset of data

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    X : NDArray[np.float64],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float64],(sum(p),)
        The observation locations

    weights : NDArray[np.float64],(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : np.float64
        Average log likelihood
    """
    weights = (
        np.ones(X.shape[0]) if weights is None else weights / np.mean(weights)
    )

    return np.mean(weights * _marginal_score_samples(covariance, X, idxs))


def _marginal_score_sherman_woodbury(
    Lambda: NDArray[np.float64],
    W: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> np.float64:
    """
    Returns the marginal log-likelihood of a subset of data
    given that Sigma = WWT + Lambda

    Parameters
    ----------
    Lambda : NDArray[np.float64],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float64],(L,p)
        The low rank component

    X : NDArray[np.float64],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float64],(sum(p),)
        The observation locations

    weights : NDArray[np.float64] | None,(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : np.float64
        Average log likelihood
    """
    if weights is None:
        weights = np.ones(X.shape[0])

    return np.mean(
        weights * _marginal_score_samples_sherman_woodbury(Lambda, W, X, idxs)
    )


def _marginal_score_samples(
    covariance: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Returns the marginal log-likelihood of a subset of data
    per window

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    X : NDArray[np.float64],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float64],(sum(p),)
        The observation locations

    Returns
    -------
    scores : np.float64
        Average log likelihood
    """
    cov1 = covariance[idxs == 1]
    cov_sub = cov1[:, idxs == 1]

    return _score_samples(cov_sub, X)


def _marginal_score_samples_sherman_woodbury(
    Lambda: NDArray[np.float64],
    W: NDArray[np.float64],
    X: NDArray[np.float64],
    idxs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Returns the marginal log-likelihood of a subset of data
    per window given that Sigma = WWT + Lambda

    Parameters
    ----------
    Lambda : NDArray[np.float64],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float64],(L,p)
        The low rank component

    X : NDArray[np.float64],(N,sum(idxs))
        The centered data

    idxs: NDArray[np.float64],(sum(p),)
        The observation locations

    Returns
    -------
    scores : np.float64
        Average log likelihood
    """
    Lambda_sub = Lambda[np.ix_(idxs == 1, idxs == 1)]
    W_sub = W[:, idxs == 1]

    return _score_samples_sherman_woodbury(Lambda_sub, W_sub, X)


def _score(
    covariance: NDArray[np.float64],
    X: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> np.float64:
    """
    Returns the average log liklihood of data.

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    X : NDArray[np.float64],(N,sum(p))
        The centered data

    weights : NDArray[np.float64] | None,(N,),default=None
        The optional weights on the samples

    Returns
    -------
    avg_score : np.float64
        Average log likelihood
    """
    weights = (
        np.ones(X.shape[0]) if weights is None else weights / np.mean(weights)
    )

    return np.mean(weights * _score_samples(covariance, X))


def _score_sherman_woodbury(
    Lambda: NDArray[np.float64],
    W: NDArray[np.float64],
    X: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
) -> np.float64:
    """
    Calculate the average log likelihood of the data given a specific covariance structure.

    The covariance structure is assumed to be Sigma = WW^T + Lambda, where
    WW^T is the low-rank component and Lambda is a diagonal matrix (noise).

    Parameters
    ----------
    Lambda : NDArray[np.float64]
        The diagonal noise matrix, with shape (p, p), where 'p' represents the number of features.

    W : NDArray[np.float64]
        The low rank component matrix, with shape (L, p), where 'L' represents the number of factors
        and 'p' is the number of features.

    X : NDArray[np.float64]
        The centered data matrix, with shape (N, p), where 'N' is the number of samples and 'p' is
        the number of features. It is assumed that the data has already been centered.

    weights : NDArray[np.float64] | None, optional
        The weights for the samples, with shape (N,), where 'N' is the number of samples. If None,
        all samples are weighted equally. Default is None.

    Returns
    -------
    avg_score : np.float64
        The average log likelihood of the data under the given model parameters.

    Notes
    -----
    The function _score_samples_sherman_woodbury should be defined elsewhere and should compute
    the log likelihood of the samples given the model parameters.

    The likelihood is weighted by the `weights` parameter if provided; otherwise, each sample's
    likelihood contributes equally to the average.
    """
    if weights is None:
        weights = np.ones(X.shape[0])

    return np.mean(weights * _score_samples_sherman_woodbury(Lambda, W, X))


def _score_samples(
    covariance: NDArray[np.float64], X: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Return the log likelihood of each sample

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    X : NDArray[np.float64],(N,sum(p))
        The centered data

    Returns
    -------
    scores : np.float64
        Log likelihood for each sample
    """
    p = covariance.shape[0]
    term1 = -p / 2 * np.log(2 * np.pi)

    _, logdet = la.slogdet(covariance)
    term2 = -0.5 * logdet

    quad_init = la.solve(covariance, np.transpose(X))
    difference = X * np.transpose(quad_init)
    term3 = np.sum(difference, axis=1)

    return term1 + term2 - 0.5 * term3


def _score_samples_sherman_woodbury(
    Lambda: NDArray[np.float64], W: NDArray[np.float64], X: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Return the log likelihood of each sample

    Parameters
    ----------
    Lambda : NDArray[np.float64],(p,p)
        The diagonal noise matrix

    W : NDArray[np.float64],(L,p)
        The low rank component

    X : NDArray[np.float64],(N,sum(p))
        The centered data

    Returns
    -------
    scores : np.float64
        Log likelihood for each sample
    """
    p = Lambda.shape[1]
    I_L = np.eye(W.shape[0])
    term1 = -p / 2 * np.log(2 * np.pi)
    term2 = -0.5 * ldet_sherman_woodbury_fa(Lambda, W)

    Li = la.inv(Lambda)
    C = np.dot(Li, W.T)
    CtW = np.dot(C.T, W.T)
    middle = I_L + CtW
    end = la.solve(middle, C.T)  # (I + WLiW^T)^{-1}WLi
    prod = np.dot(C, end)
    Lip = Li - prod
    quad_init = np.dot(Lip, X.T)
    difference = X * np.transpose(quad_init)
    term3 = np.sum(difference, axis=1)

    return term1 + term2 - 0.5 * term3


def _entropy(covariance: NDArray[np.float64]) -> np.float64:
    """
    Computes the entropy of a Gaussian distribution parameterized by
    covariance.

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    Returns
    -------
    entropy : np.float64
        The differential entropy of the distribution
    """
    cov_new = 2 * np.pi * np.e * covariance
    _, logdet = la.slogdet(cov_new)

    return 0.5 * logdet


def _entropy_subset(
    covariance: NDArray[np.float64], idxs: NDArray[np.float64]
) -> np.float64:
    """
    Computes the entropy of a subset of the Gaussian distribution
    parameterized by covariance.

    Parameters
    ----------
    covariance : NDArray[np.float64],(p,p)
        The covariance matrix

    idxs: NDArray[np.float64],(sum(p),)
        The observation locations

    Returns
    -------
    entropy : np.float64
        The differential entropy of the distribution
    """
    cov1 = covariance[idxs == 1]
    cov_sub = cov1[:, idxs == 1]
    entropy = _entropy(cov_sub)

    return entropy


def _mutual_information(
    covariance: NDArray[np.float64],
    idxs1: NDArray[np.float64],
    idxs2: NDArray[np.float64],
) -> np.float64:
    """
    Compute the mutual information between two groups of features.

    Uses the Gaussian entropy formula: ``I(X1; X2) = H(X1) - H(X1|X2)``.

    Parameters
    ----------
    covariance : ndarray of shape (n_features, n_features)
        The covariance matrix.

    idxs1 : ndarray of shape (n_features,)
        Boolean or integer index array selecting the first group of features.

    idxs2 : ndarray of shape (n_features,)
        Boolean or integer index array selecting the second group of features.

    Returns
    -------
    mutual_information : np.float64
        Mutual information between the two feature groups in nats.
    """
    idxs = np.logical_or(idxs1, idxs2).astype(int)
    cov_sub = covariance[np.ix_(idxs == 1, idxs == 1)]
    idxs1_sub = idxs1[idxs == 1]
    Hy = _entropy(cov_sub[np.ix_(idxs1_sub == 1, idxs1_sub == 1)])

    _, covariance_conditional = _get_conditional_parameters(
        cov_sub, 1 - idxs1_sub
    )
    H_y_given_x = _entropy(covariance_conditional)
    mutual_information = Hy - H_y_given_x

    return mutual_information


def ldet_sherman_woodbury_fa(Lambda: NDArray, W: NDArray) -> float:
    """
    This converts the log determinant of a matrix Lambda + W^TW where
    Lambda is diagonal. Fa for factor analysis

    Parameters
    ----------
    W : np.array(n_components,p)
        The PCA loadings matrix

    Lambda : NDArray,(p,p)
        An easy to invert (diagonal) noise matrix

    Returns
    -------
    log_determinant : float
        The log determinant of the covariance matrix
    """
    _, ldetL = la.slogdet(Lambda)
    LiW = la.solve(Lambda, W.T)
    WtLiW = np.dot(W, LiW)
    IWtLiW = np.eye(W.shape[0]) + WtLiW
    _, ldetP = la.slogdet(IWtLiW)
    log_determinant = ldetP + ldetL
    return log_determinant


def ldet_sherman_woodbury_full(
    A: NDArray, U: NDArray, B: NDArray, V: NDArray
) -> float:
    """
    The value log|A+ UBV|. Useful when A and B are simple to invert and UBV
    is low rank

    Parameters
    ----------
    A : NDArray,(p,p)
        The first square matrix

    U :  NDArray,(p,L)
        The wide matrix

    B : NDArray,(L,L)
        The tiny square matrix

    V : NDArray,(L,p)
        Tall matrix

    Returns
    -------
    log_determinant : float
        The log determinant of the covariance matrix
    """
    _, ldetA = la.slogdet(A)
    _, ldetB = la.slogdet(B)
    term2 = np.dot(V, la.solve(A, U))
    term1 = la.inv(B)
    _, ldetProd = la.slogdet(term1 + term2)
    log_determinant = ldetA + ldetB + ldetProd
    return log_determinant


def inv_sherman_woodbury_fa(Lambda: NDArray, W: NDArray) -> NDArray[np.float64]:
    """
    This converts the inverse of a matrix Lambda + W^TW where
    Lambda is diagonal. Fa for factor analysis

    Parameters
    ----------
    W : np.array(n_components,p)
        The PCA loadings matrix

    Lambda : NDArray,(p,p)
        An easy to invert (diagonal) noise matrix

    Returns
    -------
    Sigma_inv : NDArray,(p,p)
        The inverse of the covariance matrix
    """
    I_L = np.eye(W.shape[0])
    I_p = np.eye(W.shape[1])
    Lambda_inv = la.inv(Lambda)
    WLi = np.dot(W, Lambda_inv)
    inner = I_L + np.dot(WLi, W.T)
    inner_inv = la.inv(inner)
    end = np.dot(inner_inv, WLi)
    term2 = np.dot(W.T, end)
    Imterm2 = I_p - term2
    Sigma_inv = np.dot(Lambda_inv, Imterm2)
    return Sigma_inv


def inv_sherman_woodbury_full(
    A: NDArray, U: NDArray, B: NDArray, V: NDArray
) -> NDArray[np.float64]:
    """
    This converts the inverse of a matrix (A + UBV)

    Parameters
    ----------
    A : NDArray,(p,p)
        The first square matrix

    U :  NDArray,(p,L)
        The wide matrix

    B : NDArray,(L,L)
        The tiny square matrix

    V : NDArray,(L,p)
        Tall matrix

    Returns
    -------
    Sigma_inv : NDArray,(p,p)
        The inverse of the matrix (A + UBV)
    """
    Ainv = la.inv(A)  # Needed explicitly anyways
    AiU = np.dot(Ainv, U)
    Binv = la.inv(B)  # Should be easy to invert
    VAinv = np.dot(V, Ainv)
    VAiU = np.dot(VAinv, U)
    middle = Binv + VAiU
    end = la.solve(middle, VAinv)
    second_term = np.dot(AiU, end)
    first_term = Ainv
    Sigma_inv = first_term - second_term
    return Sigma_inv
