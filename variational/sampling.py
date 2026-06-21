"""Monte Carlo sampling for the variational formula (Chapter 6).

  desirability_scores  — r(i) = exp(-C(X_i)/alpha)                     (Eq. 6.25)
  free_energy_estimate — F(Q*) = -alpha * log( (1/N) sum r(i) )        (Eq. 6.27)
  inverse_cdf_resample — sample from Q* via inverse CDF (Algorithm 10)
"""

import numpy as np


def desirability_scores(C_values, alpha, xp=np):
    """Compute desirability scores r(i) = exp(-C(X_i) / alpha).  (Eq. 6.25)

    Uses min-subtraction for numerical stability: compute
    exp(-(C_i - C_min)/alpha) to avoid underflow when alpha is small.

    Parameters
    ----------
    C_values : array of shape (N,)
        Cost C(X_i) for each sample i = 1, ..., N.
    alpha : float
        Temperature parameter (alpha > 0).
    xp : module
        Array backend (numpy or cupy).

    Returns
    -------
    r : array of shape (N,)
        Desirability scores (unnormalized).
    C_min : float
        The minimum cost subtracted for stability. Needed to recover
        the true free energy: F(Q*) = -alpha * log(mean(r)) + C_min.
    """
    # ##########################################################
    # TODO: Compute desirability scores r(i) for each sample
    # (Eq. 6.25). Subtract C_min before exponentiating to
    # avoid numerical underflow.
    #
    # ##########################################################
    
    # raise NotImplementedError("TODO: desirability_scores")

    C_min = xp.min(C_values)
    r = xp.exp(-(C_values - C_min) / alpha)
    return r, C_min


def free_energy_estimate(C_values, alpha, xp=np):
    """Estimate free energy F(Q*) by Monte Carlo.  (Eq. 6.27)

    F(Q*) = -alpha * log( (1/N) * sum_i exp(-C(X_i)/alpha) )

    Parameters
    ----------
    C_values : array of shape (N,)
    alpha : float
    xp : module

    Returns
    -------
    F : float
        Free energy estimate.
    """
    # ##########################################################
    # TODO: Estimate free energy F(Q*) from Monte Carlo
    # samples (Eq. 6.27). Use desirability_scores() and
    # account for the C_min offset.
    #
    # ##########################################################
    
    # raise NotImplementedError("TODO: free_energy_estimate")

    r, C_min = desirability_scores(C_values, alpha, xp)
    return -alpha * float(xp.log(xp.mean(r))) + C_min


def free_energy_convergence(C_values, alpha, xp=np):
    """Running free energy estimate as a function of sample size.

    Returns an array of length N where entry k is the estimate
    using the first k+1 samples. Used for convergence plots (Fig 6.3).

    Parameters
    ----------
    C_values : array of shape (N,)
    alpha : float
    xp : module

    Returns
    -------
    F_running : array of shape (N,)
    """
    r, C_min = desirability_scores(C_values, alpha, xp)
    cumsum = xp.cumsum(r)
    counts = xp.arange(1, len(r) + 1, dtype=r.dtype)
    running_mean = cumsum / counts
    running_mean = xp.clip(running_mean, 1e-300, None)
    return -alpha * xp.log(running_mean) + C_min


def inverse_cdf_resample(samples, rewards, n_resample=1, rng=None, xp=np):
    """Algorithm 10: sample from Q* via inverse-CDF resampling.

    Given N samples X(i) drawn from the reference R, and their
    desirability scores r(i), draw new samples from Q* by:
      1. Build cumulative sum F(k) = sum_{i=1}^{k} r(i)
      2. Draw d ~ Uniform[0, r_total]
      3. Find index i' such that F(i'-1) < d <= F(i')
      4. Return X(i')

    Parameters
    ----------
    samples : array of shape (N, ...) or (N,)
        Original samples X(i) drawn from R.
    rewards : array of shape (N,)
        Desirability scores r(i) for each sample.
    n_resample : int
        Number of new samples to draw from Q*.
    rng : numpy.random.Generator or None
        Random number generator. If None, uses default.
    xp : module
        Array backend.

    Returns
    -------
    resampled : array of shape (n_resample, ...) or (n_resample,)
        Samples drawn from Q*.
    indices : array of shape (n_resample,)
        Indices into the original sample array.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ##########################################################
    # TODO: Resample from Q* via inverse CDF (Algorithm 10).
    # 1. Build cumulative sum F(k) = sum_{i=1}^{k} r(i)
    # 2. Draw n_resample uniform random values d in [0, F(N)]
    # 3. For each d, find the index i such that F(i-1) < d <= F(i)
    #    (use xp.searchsorted)
    # 4. Clip indices to valid range
    #
    # ##########################################################
    
    # raise NotImplementedError("TODO: inverse_cdf_resample")

    if rng is None:
        rng = np.random.default_rng()

    cdf = xp.cumsum(rewards)
    r_total = cdf[-1]

    if hasattr(r_total, 'item'):
        r_total_scalar = r_total.item()
    else:
        r_total_scalar = float(r_total)

    d = rng.uniform(0.0, r_total_scalar, size=n_resample)
    d = xp.asarray(d)

    indices = xp.searchsorted(cdf, d, side='left')
    indices = xp.clip(indices, 0, len(rewards) - 1)

    if hasattr(indices, 'get'):
        indices_np = indices.get()
    else:
        indices_np = indices

    resampled = samples[indices_np]
    return resampled, indices_np
