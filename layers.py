"""NumPy layers. Each layer caches what it needs in forward, returns dx in backward"""
import numpy as np


class Parameter:
    """A trainable tensor with a gradient buffer"""
    __slots__ = ("value", "grad")

    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad = np.zeros_like(value)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class Module:
    """Base class. Subclasses implement forward/backward; parameters() walks attributes"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        out: list[Parameter] = []
        seen: set[int] = set()

        def visit(obj):
            if id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, Parameter):
                out.append(obj)
            elif isinstance(obj, Module):
                for v in vars(obj).values():
                    visit(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    visit(v)

        visit(self)
        return out


def im2col(x: np.ndarray, kH: int, kW: int, stride: int = 1, pad: int = 0) -> tuple[np.ndarray, int, int]:
    """(N,C,H,W) -> (N*Ho*Wo, C*kH*kW). Returns also (Ho, Wo)"""
    N, C, H, W = x.shape
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Ho = (x.shape[2] - kH) // stride + 1
    Wo = (x.shape[3] - kW) // stride + 1
    cols = np.zeros((N, C, kH, kW, Ho, Wo), dtype=x.dtype)
    for i in range(kH):
        for j in range(kW):
            cols[:, :, i, j, :, :] = x[:, :, i:i + Ho * stride:stride, j:j + Wo * stride:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, -1), Ho, Wo


def col2im(dcol: np.ndarray, x_shape: tuple, kH: int, kW: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Inverse of im2col."""
    N, C, H, W = x_shape
    Ho = (H + 2 * pad - kH) // stride + 1
    Wo = (W + 2 * pad - kW) // stride + 1
    dcol_r = dcol.reshape(N, Ho, Wo, C, kH, kW).transpose(0, 3, 4, 5, 1, 2)
    dx = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=dcol.dtype)
    for i in range(kH):
        for j in range(kW):
            dx[:, :, i:i + Ho * stride:stride, j:j + Wo * stride:stride] += dcol_r[:, :, i, j, :, :]
    return dx[:, :, pad:H + pad, pad:W + pad] if pad else dx


class Conv2d(Module):
    """2D convolution with bias. Weights init: N(0, 0.02) per CycleGAN paper"""

    def __init__(self, in_c: int, out_c: int, k: int = 3, stride: int = 1, pad: int = 0):
        self.k, self.s, self.p = k, stride, pad
        self.W = Parameter(np.random.randn(out_c, in_c, k, k).astype(np.float32) * 0.02)
        self.b = Parameter(np.zeros(out_c, dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        col, Ho, Wo = im2col(x, self.k, self.k, self.s, self.p)
        self.col = col
        out_c = self.W.value.shape[0]
        W_flat = self.W.value.reshape(out_c, -1)
        out = col @ W_flat.T + self.b.value
        return out.reshape(x.shape[0], Ho, Wo, out_c).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        out_c = self.W.value.shape[0]
        dout_r = dout.transpose(0, 2, 3, 1).reshape(-1, out_c)
        self.W.grad += (dout_r.T @ self.col).reshape(self.W.value.shape)
        self.b.grad += dout_r.sum(0)
        dcol = dout_r @ self.W.value.reshape(out_c, -1)
        return col2im(dcol, self.x_shape, self.k, self.k, self.s, self.p)


class InstanceNorm2d(Module):
    """Per-instance, per-channel normalization with affine gamma, beta"""

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = Parameter(np.ones(num_features, dtype=np.float32))
        self.beta = Parameter(np.zeros(num_features, dtype=np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True)
        std = np.sqrt(var + self.eps)
        x_hat = (x - mu) / std
        self.x_hat, self.std = x_hat, std
        return self.gamma.value.reshape(1, -1, 1, 1) * x_hat + self.beta.value.reshape(1, -1, 1, 1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        gamma = self.gamma.value.reshape(1, -1, 1, 1)
        self.gamma.grad += (dout * self.x_hat).sum(axis=(0, 2, 3))
        self.beta.grad += dout.sum(axis=(0, 2, 3))
        dx_hat = dout * gamma
        mean1 = dx_hat.mean(axis=(2, 3), keepdims=True)
        mean2 = (dx_hat * self.x_hat).mean(axis=(2, 3), keepdims=True)
        return (dx_hat - mean1 - self.x_hat * mean2) / self.std


class ReflectionPad2d(Module):
    """Reflection padding (no learnable params). Backward sums grads back to original positions"""

    def __init__(self, pad: int):
        self.pad = pad

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        p = self.pad
        return np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="reflect")

    def backward(self, dout: np.ndarray) -> np.ndarray:
        p = self.pad
        if p == 0:
            return dout
        N, C, H, W = self.x_shape
        Hp, Wp = H + 2 * p, W + 2 * p
        # for each padded coordinate, what was the source row/col in the input
        src_i = np.concatenate([np.arange(p, 0, -1), np.arange(H), np.arange(H - 2, H - 2 - p, -1)])
        src_j = np.concatenate([np.arange(p, 0, -1), np.arange(W), np.arange(W - 2, W - 2 - p, -1)])
        tmp = np.zeros((N, C, Hp, W), dtype=dout.dtype)
        np.add.at(tmp, (slice(None), slice(None), slice(None), src_j), dout)
        dx = np.zeros((N, C, H, W), dtype=dout.dtype)
        np.add.at(dx, (slice(None), slice(None), src_i, slice(None)), tmp)
        return dx


class NearestUpsample(Module):
    """Nearest-neighbor upsampling by integer factor. Used instead of ConvTranspose to avoid checkerboard"""

    def __init__(self, factor: int = 2):
        self.f = factor

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        return x.repeat(self.f, axis=2).repeat(self.f, axis=3)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        f = self.f
        N, C, H, W = self.x_shape
        return dout.reshape(N, C, H, f, W, f).sum(axis=(3, 5))


class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


class LeakyReLU(Module):
    def __init__(self, slope: float = 0.2):
        self.slope = slope

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.where(x > 0, x, self.slope * x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * np.where(self.x > 0, 1.0, self.slope).astype(dout.dtype)


class Tanh(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = np.tanh(x)
        return self.y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (1 - self.y ** 2)


class Sequential(Module):
    """Compose modules in order"""

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for l in self.layers:
            x = l(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout


class ResidualBlock(Module):
    """Residual block: pad -> conv -> IN -> ReLU -> pad -> conv -> IN, plus skip connection"""

    def __init__(self, ch: int):
        self.body = Sequential(
            ReflectionPad2d(1),
            Conv2d(ch, ch, k=3, stride=1, pad=0),
            InstanceNorm2d(ch),
            ReLU(),
            ReflectionPad2d(1),
            Conv2d(ch, ch, k=3, stride=1, pad=0),
            InstanceNorm2d(ch),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self.body(x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout + self.body.backward(dout)
