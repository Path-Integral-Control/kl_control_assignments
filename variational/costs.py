"""Cost landscapes for variational formula examples (Chapter 6).

Each function maps points in a bounded domain to a scalar cost C(x).
The variational formula (eq. 6.1) connects these costs to the Boltzmann
distribution Q*(x) = R(x) exp(-C(x)/alpha) / Z.
"""

import numpy as np


def landscape_2d(x, y, xp=np):
    """Cost landscape on [0,1]^2.

    Bowl with minimum near center and ripple structure.
    Q* concentrates near the center where cost is lowest.
    """
    cx, cy = 0.5, 0.5

    bowl = 8.0 * ((x - cx)**2 + (y - cy)**2)

    ripple = 0.8 * xp.sin(3.0 * np.pi * x) * xp.sin(3.0 * np.pi * y)

    return bowl + ripple + 0.5


def double_well_1d(x, xp=np):
    """Double-well potential C(x) = (x^2 - 1)^2 on [-2, 2].

    Two symmetric minima at x = +/-1 (C=0), one local maximum at x=0 (C=1).
    At small alpha, Q* concentrates on both wells. At alpha -> 0, Q*
    collapses to deltas at the two global minima.
    """
    return (x**2 - 1.0)**2


def himmelblau(x, y, xp=np):
    """Himmelblau's function: four equal global minima.

    C(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    Domain typically [-5, 5]^2. Four minima at approximately:
      (3.0, 2.0), (-2.81, 3.13), (-3.78, -3.28), (3.58, -1.85)
    At moderate alpha, Q* has four equal peaks.
    """
    return (x**2 + y - 11.0)**2 + (x + y**2 - 7.0)**2
