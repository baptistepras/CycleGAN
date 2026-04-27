"""Adam optimizer"""
import numpy as np
from layers import Parameter


class Adam:
    """Adam with (b1, b2) = (0.5, 0.999) by default"""

    def __init__(self, params: list[Parameter], lr: float = 2e-4,
                 betas: tuple[float, float] = (0.5, 0.999), eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]

    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            m[:] = self.b1 * m + (1 - self.b1) * p.grad
            v[:] = self.b2 * v + (1 - self.b2) * (p.grad ** 2)
            p.value -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()
