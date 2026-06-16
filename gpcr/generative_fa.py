from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA  # type: ignore
from torch import Tensor, nn
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange

from ._base import BasePCASGDModel
from ._misc_np import softplus_inverse_np
from ._sherman_woodbury_pt import mvn_log_prob_sw


class PPCA(BasePCASGDModel):
    def __init__(
        self,
        n_components: int = 2,
        prior_options: dict | None = None,
        training_options: dict | None = None,
    ):
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()

        self.W_: NDArray[np.float_] | None = None
        self.sigmas_: NDArray[np.float_] | None = None
        self.sigma2_: np.float_ | None = None
        self.p: int | None = None

    def __repr__(self):
        return f"PPCApt(n_components={self.n_components})"

    def fit(
        self,
        X: NDArray[np.float_],
        progress_bar: bool = True,
        seed: int = 2021,
        sherman_woodbury: bool = False,
    ) -> "PPCA":
        """
        Fit the PPCA model to data using stochastic gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        progress_bar : bool, default=True
            Whether to display a tqdm progress bar during training.

        seed : int, default=2021
            Seed for the random number generator used for mini-batch sampling.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity to evaluate the
            log-likelihood, which is advantageous when ``n_features`` is
            large relative to ``n_components``.

        Returns
        -------
        self : PPCA
            Fitted estimator.
        """
        self._test_inputs(X)
        training_options = self.training_options
        N, p = X.shape
        self.p = p
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        rng = np.random.default_rng(int(seed))

        W_, sigmal_ = self._initialize_variables(device, X)

        X_tensor = self._transform_training_data(device, X)[0]

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        eye_L = torch.tensor(
            np.eye(self.n_components).astype(np.float32), device=device
        )
        zeros_p = torch.tensor(
            np.zeros(p).astype(np.float32), dtype=torch.float32, device=device
        )
        softplus = nn.Softplus()

        _prior = self._create_prior(device)

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            idx = rng.choice(
                X_tensor.shape[0],
                size=training_options["batch_size"],
                replace=False,
            )
            X_batch = X_tensor[idx].float()

            sigma = softplus(sigmal_)

            if sherman_woodbury:
                Lambda = sigma * eye
                like_tot = mvn_log_prob_sw(X_batch, zeros_p, Lambda, W_, eye_L)
            else:
                WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
                Sigma = WWT + sigma * eye
                m = MultivariateNormal(zeros_p, Sigma)
                like_tot = torch.mean(m.log_prob(X_batch))

            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_tot, like_prior, posterior)

        self._store_instance_variables(device, trainable_variables)

        return self

    def get_covariance(self):
        """
        Compute the model covariance matrix.

        The covariance is given by ``Sigma = W^T W + sigma^2 * I``.

        Returns
        -------
        covariance : ndarray of shape (n_features, n_features)
            The PPCA covariance matrix.
        """
        if self.W_ is None or self.sigma2_ is None or self.p is None:
            raise ValueError("Fit model first")

        return np.dot(self.W_.T, self.W_) + self.sigma2_ * np.eye(self.p)

    def get_noise(self):
        """
        Return the observational noise as a diagonal matrix.

        For PPCA the noise is isotropic: ``Lambda = sigma^2 * I``.

        Returns
        -------
        Lambda : ndarray of shape (n_features, n_features)
            The isotropic diagonal noise matrix.
        """
        if self.sigma2_ is None or self.p is None:
            raise ValueError("Fit model first")

        return self.sigma2_ * np.eye(self.p)

    def _create_prior(self, device):
        """
        Construct the log-prior function for PPCA parameters.

        Creates a closure over ``prior_options`` that computes the
        log-prior on ``W_`` (Gaussian regularization) and ``sigma``
        (Gamma prior on the noise variance).

        Parameters
        ----------
        device : Any
            PyTorch device (CPU or CUDA) on which to place prior tensors.

        Returns
        -------
        log_prior : callable
            Function that accepts the list of trainable variable tensors
            and returns a scalar Tensor of the log-prior value.
        """
        prior_options = self.prior_options

        def log_prior(trainable_variables: list[Tensor]):
            W_ = trainable_variables[0]
            sigmal_ = trainable_variables[1]
            sigma_ = nn.Softplus()(sigmal_)

            part1 = (
                -1 * prior_options["weight_W"] * torch.mean(torch.square(W_))
            )
            part2 = (
                Gamma(prior_options["alpha"], prior_options["beta"])
                .log_prob(sigma_)
                .to(device)
            )
            out = torch.mean(part1 + part2)
            return out

        return log_prior

    def _initialize_variables(
        self, device: Any, X: NDArray[np.float_]
    ) -> tuple[Tensor, Tensor]:
        """
        Initialize model parameters from a sklearn PCA warm start.

        Fits a PCA model to ``X``, uses its loadings as ``W_``, and sets
        the initial noise ``sigma^2`` to the mean squared reconstruction
        error.

        Parameters
        ----------
        device : Any
            PyTorch device (CPU or CUDA) on which to place the tensors.

        X : ndarray of shape (n_samples, n_features)
            The training data.

        Returns
        -------
        W_ : Tensor of shape (n_components, n_features)
            Initial loadings tensor with ``requires_grad=True``.

        sigmal_ : Tensor of shape (1,)
            Initial unrectified noise variance with ``requires_grad=True``.
        """
        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True, device=device)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1).astype(np.float32))
        sigmal_ = torch.tensor(sinv, requires_grad=True, device=device)
        return W_, sigmal_

    def _store_instance_variables(  # type: ignore[override]
        self, device: Any, trainable_variables: list[Tensor]
    ) -> None:
        """
        Store the learned parameters as numpy arrays on the instance.

        Parameters
        ----------
        device : Any
            PyTorch device used during training.

        trainable_variables : list of Tensor
            Tensors ``[W_, sigmal_]`` after optimization.

        Attributes
        ----------
        W_ : ndarray of shape (n_components, n_features)
            Fitted factor loadings.

        sigma2_ : np.float_
            Fitted isotropic noise variance.
        """
        if device.type == "cuda":
            self.W_ = trainable_variables[0].detach().cpu().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().cpu().numpy()
            )
        else:
            self.W_ = trainable_variables[0].detach().numpy()
            self.sigma2_ = (
                nn.Softplus()(trainable_variables[1]).detach().numpy()
            )

    def _test_inputs(self, X: NDArray[np.float_]) -> None:
        """
        Validate inputs before fitting.

        Parameters
        ----------
        X : ndarray
            The training data.

        Raises
        ------
        ValueError
            If ``X`` is not a numpy array or the batch size exceeds the
            number of samples.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")

    def _fill_prior_options(
        self, prior_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fill in default prior options for PPCA parameters.

        Parameters
        ----------
        prior_options : dict
            User-supplied prior options. Recognized keys:

            - ``weight_W`` : float, default=0.01
                L2 regularization weight on the loadings ``W``.
            - ``alpha`` : float, default=3.0
                Shape parameter of the Gamma prior on the noise variance.
            - ``beta`` : float, default=3.0
                Rate parameter of the Gamma prior on the noise variance.

        Returns
        -------
        options : dict
            Merged dictionary of prior options with defaults applied.
        """
        default_dict = {"weight_W": 0.01, "alpha": 3.0, "beta": 3.0}

        return {**default_dict, **prior_options}


class FactorAnalysis(BasePCASGDModel):
    def __init__(
        self,
        n_components=2,
        prior_options: dict | None = None,
        training_options: dict | None = None,
    ):
        """
        Factor analysis with heterogeneous diagonal noise per feature.

        Each feature is given its own noise variance, unlike PPCA which
        uses a single isotropic noise parameter.

        Parameters
        ----------
        n_components : int, default=2
            The dimensionality of the latent space.

        training_options : dict, default=None
            Options for gradient descent passed to ``_fill_training_options``.

        prior_options : dict, default=None
            Options for priors on model parameters passed to
            ``_fill_prior_options``.
        """
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()
        self.p: int | None = None
        self.W_: NDArray[np.float_] | None = None
        self.sigmas_: NDArray[np.float_] | None = None

    def __repr__(self) -> str:
        return f"FactorAnalysispt(n_components={self.n_components})"

    def fit(
        self,
        X: NDArray[np.float_],
        progress_bar: bool = True,
        seed: int = 2021,
        sherman_woodbury: bool = False,
    ) -> "FactorAnalysis":
        """
        Fit the factor analysis model to data using stochastic gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The centered data matrix.

        progress_bar : bool, default=True
            Whether to display a tqdm progress bar during training.

        seed : int, default=2021
            Seed for the random number generator used for mini-batch sampling.

        sherman_woodbury : bool, default=False
            If ``True``, uses the Sherman-Woodbury identity to evaluate the
            log-likelihood.

        Returns
        -------
        self : FactorAnalysis
            Fitted estimator.
        """
        self._test_inputs(X)
        training_options = self.training_options
        device = torch.device(
            "cuda"
            if torch.cuda.is_available() and training_options["use_gpu"]
            else "cpu"
        )

        N, p = X.shape
        self.p = p

        rng = np.random.default_rng(int(seed))

        W_, sigmal_ = self._initialize_variables(device, X)

        X_ = self._transform_training_data(device, X)[0]

        trainable_variables = [W_, sigmal_]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )
        softplus = nn.Softplus()

        _prior = self._create_prior(device)

        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        zeros_p = torch.zeros(p, device=device)

        for i in trange(
            training_options["n_iterations"], disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]

            sigmas = softplus(sigmal_)
            Lambda = torch.diag(sigmas)

            if sherman_woodbury:
                like_tot = mvn_log_prob_sw(X_batch, zeros_p, Lambda, W_, eye)
            else:
                WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
                Sigma = WWT + Lambda
                m = MultivariateNormal(zeros_p, Sigma)
                like_tot = torch.mean(m.log_prob(X_batch))
            like_prior = _prior(trainable_variables)
            posterior = like_tot + like_prior / N
            loss = -1 * posterior

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self._save_losses(i, device, like_tot, like_prior, posterior)

        self._store_instance_variables(device, trainable_variables)

        return self

    def get_covariance(self) -> NDArray[np.float_]:
        """
        Compute the model covariance matrix.

        The covariance is given by ``Sigma = W^T W + diag(sigmas)``.

        Returns
        -------
        covariance : ndarray of shape (n_features, n_features)
            The factor analysis covariance matrix.
        """
        if self.W_ is None or self.sigmas_ is None:
            raise ValueError("Fit model first")

        return np.dot(self.W_.T, self.W_) + np.diag(self.sigmas_)

    def get_noise(self) -> NDArray[np.float_]:
        """
        Return the observational noise as a diagonal matrix.

        Returns
        -------
        Lambda : ndarray of shape (n_features, n_features)
            Diagonal noise matrix ``diag(sigmas_)``.
        """
        if self.sigmas_ is None:
            raise ValueError("Fit model first")

        return np.diag(self.sigmas_)

    def _create_prior(self, device) -> Callable[[list[Tensor]], Tensor]:
        """
        Construct the log-prior function for factor analysis parameters.

        Creates a closure that computes a Gamma log-prior on the
        (softmax-transformed) per-feature noise variances.

        Parameters
        ----------
        device : Any
            PyTorch device (CPU or CUDA) on which to place prior tensors.

        Returns
        -------
        log_prior : callable
            Function that accepts the list of trainable variable tensors
            and returns a scalar Tensor of the log-prior value.
        """

        def log_prior(trainable_variables: list[Tensor]) -> Tensor:
            sigma_ = nn.Softmax()(trainable_variables[1])
            return torch.mean(
                Gamma(self.prior_options["alpha"], self.prior_options["beta"])
                .log_prob(sigma_)
                .to(device)
            )

        return log_prior

    def _fill_prior_options(
        self, prior_options: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Fill in default prior options for factor analysis parameters.

        Parameters
        ----------
        prior_options : dict
            User-supplied prior options. Recognized keys:

            - ``alpha`` : float, default=3.0
                Shape parameter of the Gamma prior on noise variances.
            - ``beta`` : float, default=3.0
                Rate parameter of the Gamma prior on noise variances.

        Returns
        -------
        options : dict
            Merged dictionary of prior options with defaults applied.
        """
        default_dict = {"alpha": 3.0, "beta": 3.0}
        return {**default_dict, **prior_options}

    def _initialize_variables(self, device: Any, X: NDArray[np.float_]):
        """
        Initialize model parameters from a sklearn PCA warm start.

        Fits a PCA model to ``X``, uses its loadings as ``W_``, and
        initializes per-feature noise variances to the mean squared
        reconstruction error.

        Parameters
        ----------
        device : Any
            PyTorch device (CPU or CUDA) on which to place the tensors.

        X : ndarray of shape (n_samples, n_features)
            The training data.

        Returns
        -------
        W_ : Tensor of shape (n_components, n_features)
            Initial loadings tensor with ``requires_grad=True``.

        sigmal_ : Tensor of shape (n_features,)
            Initial unrectified per-feature noise variances with
            ``requires_grad=True``.
        """
        if self.p is None:
            raise ValueError("Fit model first")

        model = PCA(self.n_components)
        S_hat = model.fit_transform(X)
        W_init = model.components_.astype(np.float32)
        W_ = torch.tensor(W_init, requires_grad=True, device=device)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = torch.tensor(
            sinv[0] * np.ones(self.p).astype(np.float32),
            requires_grad=True,
            device=device,
        )

        return W_, sigmal_

    def _store_instance_variables(  # type: ignore[override]
        self, device: Any, trainable_variables: list[Tensor]
    ) -> None:
        """
        Store the learned parameters as numpy arrays on the instance.

        Parameters
        ----------
        device : Any
            PyTorch device used during training.

        trainable_variables : list of Tensor
            Tensors ``[W_, sigmal_]`` after optimization.

        Attributes
        ----------
        W_ : ndarray of shape (n_components, n_features)
            Fitted factor loadings.

        sigmas_ : ndarray of shape (n_features,)
            Fitted per-feature noise standard deviations (after softplus).
        """
        if device.type == "cuda":
            self.W_ = trainable_variables[0].detach().cpu().numpy()
            self.sigmas_ = (
                nn.Softplus()(trainable_variables[1]).detach().cpu().numpy()
            )
        else:
            self.W_ = trainable_variables[0].detach().numpy()
            self.sigmas_ = (
                nn.Softplus()(trainable_variables[1]).detach().numpy()
            )

    def _test_inputs(self, X: NDArray[np.float_]):
        """
        Validate inputs before fitting.

        Parameters
        ----------
        X : ndarray
            The training data.

        Raises
        ------
        ValueError
            If ``X`` is not a numpy array or the batch size exceeds the
            number of samples.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data is numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
