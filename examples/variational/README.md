# examples/variational/ — Variational Formula Examples (Chapter 6)

Two examples demonstrating the variational formula and Boltzmann sampling. No dynamics are involved. These are static importance sampling problems over distributions.

## boltzmann_sampling.py

Implements Algorithm 10 (inverse-CDF resampling from Q\*) on a 2D cost landscape C(x,y) defined on [0,1]^2.

Plots (2x2):
1. 3D surface of C(x,y)
2. Uniform samples from R colored by desirability score
3. Resampled points from Q\*, concentrated in low-cost regions
4. Free energy F(Q\*) convergence as N grows

```bash
python3 examples/variational/boltzmann_sampling.py
python3 examples/variational/boltzmann_sampling.py --alpha 0.5 --N 1000
python3 examples/variational/boltzmann_sampling.py --benchmark          # CPU timing
python3 examples/variational/boltzmann_sampling.py --benchmark --gpu    # GPU timing
```

## alpha_sweep.py

Sweeps the temperature parameter alpha on the 1D double-well potential C(x) = (x^2 - 1)^2. At small alpha, Q\* concentrates on both minima (x = +/-1). At large alpha, Q\* flattens toward the uniform reference R.

Plots:
- Top row: 5 panels, one per alpha value. Each shows C(x), the exact Q\*(x), and a histogram of resampled points.
- Bottom: F(Q\*) vs alpha, analytic curve and Monte Carlo estimates.

```bash
python3 examples/variational/alpha_sweep.py
python3 examples/variational/alpha_sweep.py --N 5000
```
