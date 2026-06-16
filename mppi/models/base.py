"""Abstract base class for MPPI dynamics models."""

from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    """Base class for all dynamics models used with MPPI.

    All methods that operate on state or control arrays expect batched inputs.
    State x has shape (K, state_dim), control u has shape (K, control_dim),
    where K is the number of rollout samples.

    The `xp` parameter is the array backend module (numpy or cupy), passed
    explicitly so models work on both CPU and GPU without conditional imports.
    """

    @property
    @abstractmethod
    def state_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def control_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def control_bounds(self) -> tuple:
        """Return (lower, upper) arrays of shape (control_dim,)."""
        ...

    @abstractmethod
    def dynamics(self, x, u, xp):
        """Compute dx/dt = f(x, u). Returns (K, state_dim)."""
        ...

    @abstractmethod
    def step(self, x, u, dt, xp):
        """Integrate one timestep. Returns x_next of shape (K, state_dim)."""
        ...

    @abstractmethod
    def running_cost(self, x, u, t, xp):
        """Running cost at time t. Returns (K,) array of costs."""
        ...

    @abstractmethod
    def terminal_cost(self, x, xp):
        """Terminal cost. Returns (K,) array of costs."""
        ...

    def nominal_control(self, x, t, xp):
        """Optional nominal controller for warm-starting. Default: zeros."""
        return xp.zeros((x.shape[0], self.control_dim), dtype=xp.float32)

    # -- Continuous-time MPPI (SDE formulation) --

    def drift(self, x, u, xp):
        """Drift f(x,u) for SDE dx = f(x,u)dt + G(x)dW. Default: same as dynamics."""
        return self.dynamics(x, u, xp)

    def diffusion(self, x, xp):
        """Diffusion matrix G(x). Shape (K, state_dim, noise_dim). Default: None."""
        return None

    @property
    def noise_dim(self) -> int:
        """Dimension of Brownian motion for continuous-time formulation. Default: 0."""
        return 0
