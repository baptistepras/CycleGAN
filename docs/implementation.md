# Implementation

How the NumPy CycleGAN is built, file by file. Commands are in [usage.md](usage.md), and the full analysis in the [report](report.pdf).

## Layers (`layers.py`)

Every layer is a `Module` with a `forward` and a hand-written `backward`. A `Parameter` holds a value and its gradient, and `Module.parameters()` collects them by walking the attributes, so the optimizer sees every weight without manual bookkeeping.

| Layer | Notes |
| --- | --- |
| `Conv2d` | Convolution with stride and padding. `im2col` unfolds the input patches into a matrix, so the forward pass is one matrix product, and `col2im` folds the gradients back in the backward pass. |
| `InstanceNorm2d` | Per image and per channel normalization, as in the paper. |
| `ReflectionPad2d` | Reflection padding, to avoid border artifacts. |
| `NearestUpsample` | Nearest neighbor upsampling, used with a convolution instead of a transposed convolution to avoid checkerboard artifacts. |
| `ReLU`, `LeakyReLU`, `Tanh` | Activations. |
| `Sequential`, `ResidualBlock` | Containers. The residual block adds its input to its output in both passes. |

## Architecture (`models.py`)

- **Generator.** The ResNet generator of Johnson et al.: reflection padding, a 7 x 7 convolution, two downsampling convolutions, `n_res` residual blocks (6 in our runs), two upsampling stages, and a final `tanh`. Instance norm follows every convolution except the last.
- **Discriminator.** The 70 x 70 PatchGAN of the paper: it classifies overlapping patches as real or fake, with LeakyReLU activations.

Weights are initialized from a Gaussian with mean 0 and standard deviation 0.02, as in the paper.

## Losses and training (`train.py`)

- **Adversarial loss.** LSGAN, so the discriminator regresses 1 for real and 0 for fake patches.
- **Cycle consistency.** ℓ1 between an image and its reconstruction, in both directions, weighted by 10.
- **Identity.** ℓ1 between an image and its translation by the generator that targets its own domain, weighted by 0.5 times the cycle weight.
- **Image buffer.** Each discriminator sees a mix of current fakes and fakes from a buffer of the last 50, as in the paper.
- **Optimization.** Adam with a learning rate of 2e-4 and betas (0.5, 0.999). The rate is constant until `--decay_start`, then decays linearly to zero. Discriminator gradients are halved, which matches dividing their loss by 2 in the paper.

One training step runs the two cycles, backpropagates through both generators by hand, updates them, then updates each discriminator.

## Data and evaluation

- `data.py` loads unpaired images, resizes them to `--size`, scales them to [-1, 1], and can cap the number of images per side.
- `test.py` translates test images in one or both directions, saves strips (input, translation, reconstruction), and prints the cycle and identity ℓ1 errors. With `--with_d`, it also reports the mean discriminator outputs.

## Differences from the paper

The method is the same, but the scale is much smaller: 64 x 64 images instead of 256 x 256, 150 images per side, 20 epochs instead of 200, and a CPU instead of a GPU. Upsampling uses nearest neighbor plus convolution instead of transposed convolutions.
