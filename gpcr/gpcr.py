"""
This implements Factor analysis but with variational inference to allow us
to supervise a subset of latent variables to be predictive of an auxiliary 
outcome. Currently only implements logistic regression but in the future 
will be modified to allow for more general predictive outcomes.

Note that in the code W is a L x p array, where L is the latent dimensionality
and  p is the covariate dimension, for implementation convenience. However,
mathematically it is often pxL for notational convenience. Given that the
most insidious errors are mathematical in nature rather than coding, (as faulty
math is difficult to detect in unit tests), our notation matches mathematics
rather than code, specifically when naming WWT and WTW to match the Bishop 2006
notation rather than code.


Objects
-------
GPCRvi(PPCA)
    This is PPCA with SGD but supervises a single factor to predict y.

Methods
-------
None
"""

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm import trange

from ._misc_np import softplus_inverse_np
from .generative_fa import PPCA
from .ppca_augmented import PPCAsupervisedRandomized


def _get_projection_matrix(W_: Tensor, sigma_: Tensor):
    """
    This is currently just implemented for PPCA due to nicer formula. Will
    modify for broader application later.

    Computes the parameters for p(S|X)

    Description in future to be released paper

    Parameters
    ----------
    W_ : Tensor(n_components,p)
        The loadings

    sigma_ : Tensor
        Isotropic noise

    Returns
    -------
    Proj_X : Tensor(n_components,p)
        Beta such that np.dot(Proj_X, X) = E[S|X]

    Cov : Tensor(n_components,n_components)
        Var(S|X)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_components = int(W_.shape[0])
    eye = torch.tensor(np.eye(n_components).astype(np.float32), device=device)
    M_init = torch.matmul(W_, torch.transpose(W_, 0, 1))
    M_end = sigma_ * eye
    M = M_init + M_end
    Proj_X = torch.linalg.solve(M, W_)
    Cov = torch.linalg.inv(M) * sigma_
    return Proj_X, Cov


class GPCRvi(PPCA):
    """
    This implements supervised PPCA according to the paper draft in
    prepration (Talbot et al, 2023), robust variational inference with
    variational objectivesThat is the generative mechanism matches
    probabilistic PCA with  isotropic variance. However, a variational
    lower bound ona predictive objective is used to ensure that a subset
    of latent variables are predictive of an auxiliary task.

    Parameters
    ----------
    n_components : int
        The latent dimensionality

    n_supervised : int
        The number of predictive latent variables

    prior_options : dict
        The hyperparameters for the prior on the parameters

    mu : float>0,default=1.0

    gamma : float,default=10.0

    d_weight_min : float or None, default=None
        Lower bound on each element of the predictive weight vector.
        If None, no lower bound is applied.

    d_weight_max : float or None, default=None
        Upper bound on each element of the predictive weight vector.
        If None, no upper bound is applied.

    d_bias_min : float or None, default=None
        Lower bound on the predictive bias term.
        If None, no lower bound is applied.

    d_bias_max : float or None, default=None
        Upper bound on the predictive bias term.
        If None, no upper bound is applied.

    d_freeze_iters : int, default=0
        Number of iterations to keep predictive coefficients D = (d_weights,
        d_bias) frozen at their warm-start initialization. During this phase,
        only W and sigma^2 receive gradient updates. This implements Phase 1
        of a block-coordinate-ascent strategy: the generative parameters learn
        to arrange the latent space so the *fixed* D already predicts well,
        which forces the posterior's supervised dimensions to genuinely carry
        predictive information.  Set to 0 to disable (joint optimization from
        the start, matching the original gPCR Algorithm 1).

    d_warmup_iters : int, default=0
        After the freeze phase ends, the effective learning-rate multiplier
        for D is linearly ramped from 0 to 1 over this many iterations.
        Concretely, at iteration i in the warmup window the D-gradients are
        scaled by alpha(i) = (i - d_freeze_iters) / d_warmup_iters.
        This prevents a sudden gradient shock from destabilizing the
        converged (W, sigma^2).  Set to 0 to unfreeze D instantly.

    d_anchor_weight : float, default=0.0
        Coefficient of a proximal / L2-anchoring penalty
        lambda * (||d_weights - d_weights_init||^2 + ||d_bias - d_bias_init||^2)
        that is added to the loss once D is unfrozen. The effective weight is
        annealed from d_anchor_weight down to 0 over the warmup window (or
        held constant if d_warmup_iters == 0 and d_anchor_weight > 0).
        This regularizes early D-updates toward the warm-start, acting as a
        trust region.  Set to 0.0 to disable.


    """

    def __init__(
        self,
        n_components: int = 2,
        n_supervised: int = 1,
        prior_options: dict | None = None,
        mu: float = 1.0,
        gamma: float = 10.0,
        d_weight_min: float | None = None,
        d_weight_max: float | None = None,
        d_bias_min: float | None = None,
        d_bias_max: float | None = None,
        d_freeze_iters: int = 0,
        d_warmup_iters: int = 0,
        d_anchor_weight: float = 0.0,
        training_options: dict | None = None,
    ):
        self.mu = float(mu)
        self.gamma = float(gamma)
        self.d_weight_min = d_weight_min
        self.d_weight_max = d_weight_max
        self.d_bias_min = d_bias_min
        self.d_bias_max = d_bias_max
        self.d_freeze_iters = int(d_freeze_iters)
        self.d_warmup_iters = int(d_warmup_iters)
        self.d_anchor_weight = float(d_anchor_weight)
        self.n_supervised = int(n_supervised)
        super().__init__(
            n_components=n_components,
            prior_options=prior_options,
            training_options=training_options,
        )
        self._initialize_save_losses()
        self.losses_supervision = np.empty(
            self.training_options["n_iterations"]
        )

    # override needed for mypy to ignore the non-optional `y` argument
    def fit(  # type: ignore[override]
        self,
        X: NDArray[np.float_],
        y: NDArray[np.float_],
        task: str = "classification",
        progress_bar: bool = True,
        seed: int = 2021,
    ) -> "GPCRvi":
        """
        Fits a model given covariates X as well as option labels y in the
        supervised methods

        Parameters
        ----------
        X : NDArray,(n_samples,n_covariates)
            The data

        y : NDArray,(n_samples,n_prediction)
            Covariates we wish to predict. For now lazy and assuming
            logistic regression.

        task : string,default='classification'
            Is this prediction, multinomial regression, or classification

        progress_bar : bool,default=True
            Whether to print the progress bar to monitor time

        seed : int,default=2021
            The random number generator seed used to ensure reproducibility

        Returns
        -------
        self : GPCRvi
            The model
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self._test_inputs(X, y)
        rng = np.random.default_rng(int(seed))
        training_options = self.training_options
        N, p = X.shape
        self.p = p

        W_, sigmal_ = self._initialize_variables(X, y)

        # X_, y_ = self._transform_training_data(X, 1.0 * y)
        X_ = torch.tensor(X.astype(np.float32))
        y_ = torch.tensor(y.astype(np.float32))
        X_ = X_.to(device)
        y_ = y_.to(device)

        # ---- Initialize learnable predictive coefficients D ----
        # D = (d_weights, d_bias) is the affine predictive model from the
        # supervised latent dimensions to the outcome, jointly optimized
        # with W and sigma^2 as described in gPCR Algorithm 1.
        #
        # d_weights: initialized from a warm-start logistic/linear regression
        # d_bias:    initialized from the same warm-start model's intercept
        if task == "classification":
            sigm = nn.Sigmoid()
            supervision_loss = nn.BCELoss(reduction="mean")

            mod = LogisticRegression()
            mod.fit(X, 1.0 * y)

            # Project the full-space logistic regression coefficients into
            # the supervised latent subspace to get a warm-start for d_weights.
            # W_sup is (n_supervised, p); coef is (1, p).
            W_sup_np = W_.detach().cpu().numpy()[: self.n_supervised]
            coef_projected = mod.coef_ @ W_sup_np.T  # (1, n_supervised)
            d_weights = torch.tensor(
                coef_projected.flatten().astype(np.float32),
                requires_grad=True,
                device=device,
            )
            d_bias = torch.tensor(
                mod.intercept_.astype(np.float32),
                requires_grad=True,
                device=device,
            )
        elif task == "regression":
            supervision_loss = nn.MSELoss()
            d_weights = torch.ones(
                self.n_supervised,
                requires_grad=True,
                device=device,
                dtype=torch.float32,
            )
            d_bias = torch.zeros(
                1,
                requires_grad=True,
                device=device,
                dtype=torch.float32,
            )
        else:
            err_msg = f"unrecognized_task {task}, must be regression or classification"
            raise ValueError(err_msg)

        # ---- Save initial D for proximal anchoring ----
        d_weights_init = d_weights.detach().clone()
        d_bias_init = d_bias.detach().clone()

        trainable_variables = [W_, sigmal_, d_weights, d_bias]

        optimizer = torch.optim.SGD(
            trainable_variables,
            lr=training_options["learning_rate"],
            momentum=training_options["momentum"],
        )

        eye = torch.tensor(np.eye(p).astype(np.float32), device=device)
        softplus = nn.Softplus()

        _prior = self._create_prior()

        # ---- Precompute unfreezing schedule boundaries ----
        freeze_end = self.d_freeze_iters
        warmup_end = freeze_end + self.d_warmup_iters

        for i in trange(
            int(training_options["n_iterations"]), disable=not progress_bar
        ):
            idx = rng.choice(
                X_.shape[0], size=training_options["batch_size"], replace=False
            )
            X_batch = X_[idx]
            y_batch = y_[idx]

            sigma = softplus(sigmal_)
            WWT = torch.matmul(torch.transpose(W_, 0, 1), W_)
            Sigma = WWT + sigma * eye

            like_prior = _prior(trainable_variables)

            # Generative likelihood
            m = MultivariateNormal(torch.zeros(p, device=device), Sigma)
            like_gen = torch.mean(m.log_prob(X_batch))

            # Predictive lower bound
            P_x, Cov = _get_projection_matrix(W_, sigma)
            mean_z = torch.matmul(X_batch, torch.transpose(P_x, 0, 1))

            # Reparameterization trick: eps ~ N(0, I_L)
            eps = torch.randn_like(mean_z)

            C1_2 = torch.linalg.cholesky(Cov)
            z_samples = mean_z + torch.matmul(eps, C1_2)

            z_supervised = z_samples[:, : self.n_supervised]

            if task == "regression":
                y_hat = torch.matmul(z_supervised, d_weights) + d_bias
                loss_y = supervision_loss(
                    torch.squeeze(y_hat), torch.squeeze(y_batch)
                )
            else:
                y_hat = torch.matmul(z_supervised, d_weights) + d_bias
                loss_y = supervision_loss(sigm(y_hat), y_batch)

            WTW = torch.matmul(W_, torch.transpose(W_, 0, 1))
            off_diag = WTW - torch.diag(torch.diag(WTW))
            loss_i = torch.linalg.matrix_norm(off_diag)

            posterior = like_gen + 1 / N * like_prior
            loss = -1 * posterior + self.mu * loss_y + self.gamma * loss_i

            # ---- Proximal anchoring penalty on D ----
            # Active only after the freeze phase, annealed over the warmup.
            if self.d_anchor_weight > 0.0 and i >= freeze_end:
                if self.d_warmup_iters > 0 and i < warmup_end:
                    # Linear decay from d_anchor_weight -> 0 over warmup
                    anchor_frac = 1.0 - (i - freeze_end) / self.d_warmup_iters
                else:
                    # No warmup or past warmup: no anchoring
                    anchor_frac = 0.0
                if anchor_frac > 0.0:
                    anchor_loss = (
                        anchor_frac
                        * self.d_anchor_weight
                        * (
                            torch.sum(torch.square(d_weights - d_weights_init))
                            + torch.sum(torch.square(d_bias - d_bias_init))
                        )
                    )
                    loss = loss + anchor_loss

            optimizer.zero_grad()
            loss.backward()

            # ---- Staged unfreezing: modify D-gradients before step ----
            if i < freeze_end:
                # Phase 1: D is frozen. Zero out D-gradients entirely so
                # the optimizer (including its momentum buffer) sees no
                # D-signal. W and sigma^2 train freely.
                if d_weights.grad is not None:
                    d_weights.grad.zero_()
                if d_bias.grad is not None:
                    d_bias.grad.zero_()
            elif i < warmup_end and self.d_warmup_iters > 0:
                # Phase 2 (warmup): linearly ramp D's effective learning
                # rate from 0 to 1. Scaling the gradient is equivalent to
                # a time-varying LR multiplier for these parameters only.
                alpha = (i - freeze_end) / self.d_warmup_iters
                if d_weights.grad is not None:
                    d_weights.grad.mul_(alpha)
                if d_bias.grad is not None:
                    d_bias.grad.mul_(alpha)
            # else: Phase 3 (full optimization) — no modification needed.

            optimizer.step()

            # ---- Project predictive coefficients onto feasible set ----
            with torch.no_grad():
                d_weights.clamp_(min=self.d_weight_min, max=self.d_weight_max)
                d_bias.clamp_(min=self.d_bias_min, max=self.d_bias_max)

            self._save_losses(i, like_gen, like_prior, posterior)
            self.losses_supervision[i] = loss_y.detach().cpu().numpy()

        self._store_instance_variables(trainable_variables)

        return self

    def _store_instance_variables(self, trainable_variables: list[Tensor]):
        """
        Saves the learned variables

        Parameters
        ----------
        trainable_variables : list
            List of saved variables of type Tensor

        Sets
        ----
        W_ : NDArray,(n_components,p)
            The loadings

        sigma2_ : float
            The isotropic variance

        d_weights_ : NDArray,(n_supervised,)
            The learned predictive coefficients

        d_bias_ : NDArray,(1,)
            The learned predictive bias

        """
        self.W_ = trainable_variables[0].detach().cpu().numpy()
        self.sigma2_ = (
            nn.Softplus()(trainable_variables[1]).detach().cpu().numpy()
        )
        self.d_weights_ = trainable_variables[2].detach().cpu().numpy()
        self.d_bias_ = trainable_variables[3].detach().cpu().numpy()

    def _initialize_variables(self, X: NDArray, Y: NDArray):
        """
        Initializes the variables of the model. Right now fits a PCA model
        in sklearn, uses the loadings and sets sigma^2 to be unexplained
        variance.

        Parameters
        ----------
        X : NDArray,(n_samples,p)
            The data

        Returns
        -------
        W_ : torch.tensor,shape=(n_components,p)
            The loadings of our latent factor model

        sigmal_ : torch.tensor
            The unrectified variance of the model

        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PPCAsupervisedRandomized(n_components=1, mu=100000.0)
        model.fit(X, Y.reshape((-1, 1)))
        W0 = model.W_
        S_hat = model.transform(X)
        model = PCA(self.n_components)
        model.fit(X)
        W_init = model.components_
        if np.dot(np.squeeze(S_hat), np.squeeze(Y)) > 0:
            W_init[0] = W0
        else:
            W_init[0] = -1 * W0

        S_hat = model.fit_transform(X)
        W_ = torch.tensor(W_init, requires_grad=True, device=device)
        X_recon = np.dot(S_hat, W_init)
        diff = np.mean((X - X_recon) ** 2)
        sinv = softplus_inverse_np(diff * np.ones(1))
        sigmal_ = torch.tensor(
            sinv, requires_grad=True, dtype=torch.float32, device=device
        )
        return W_, sigmal_

    def _test_inputs(self, X, y):
        """
        Just tests to make sure data is numpy array and dimensions match
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Data must be numpy array")
        if self.training_options["batch_size"] > X.shape[0]:
            raise ValueError("Batch size exceeds number of samples")
        if X.shape[0] != len(y):
            err_msg = "Length of data matrix X must equal length of labels y"
            raise ValueError(err_msg)

    def _create_prior(self):
        """
        This creates the function representing prior on pararmeters

        Parameters
        ----------
        log_prior : function
            The function representing the log density of the prior
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    def _save_losses(
        self,
        i,
        log_likelihood: Tensor,
        log_prior: NDArray[np.float_] | Tensor,
        log_posterior,
    ) -> None:
        """
        Saves the values of the losses at each iteration

        Parameters
        -----------
        i : int
            Current training iteration

        losses_likelihood : Tensor
            The log likelihood

        losses_prior : NDArray[np.float_] | Tensor
            The log prior

        losses_posterior : Tensor
            The log posterior
        """
        self.losses_likelihood[i] = log_likelihood.detach().cpu().numpy()
        if isinstance(log_prior, Tensor):
            self.losses_prior[i] = log_prior.detach().cpu().numpy()
        else:
            self.losses_prior[i] = log_prior
        self.losses_posterior[i] = log_posterior.detach().cpu().numpy()
