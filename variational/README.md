# variational/ — Boltzmann Sampling and Free Energy (Chapter 6)

Implements the variational formula (Eq. 6.1) and associated sampling procedures.

The variational formula states that for a reference distribution R, cost function C, and temperature alpha > 0:

```
min_Q { E^Q[C(X)] + alpha * D_KL(Q || R) } = -alpha * log E^R[exp(-C(X)/alpha)]
```

The minimizer is the Boltzmann distribution (Eq. 6.3):

```
Q*(x) = R(x) * exp(-C(x) / alpha) / Z
```

where Z is the normalizing constant. Chapters 7-9 apply this formula to progressively richer spaces (finite actions, trajectories, continuous-time paths).

The `sampling` module provides desirability score computation (Eq. 6.25), free energy estimation (Eq. 6.27), and inverse-CDF resampling from Q* (Algorithm 10). The `costs` module defines cost landscapes used by the examples: a 2D bowl with ripple structure on [0,1]^2 and the 1D double-well potential C(x) = (x^2-1)^2.
