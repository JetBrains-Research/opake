# ruff: noqa: INP001
"""Runnable source for the documentation quickstart."""

import torch
import torch.nn as nn
import torchopt
from torch.utils.data import DataLoader, TensorDataset

import opake.accounting as acc
import opake.dpsgd.accounting as dpsgd_acc
from opake.accounting import Accountant
from opake.dpsgd.clipping import clipped_grad
from opake.dpsgd.noise import gaussian_noise
from opake.dpsgd.sampling import PoissonSampler
from opake.functional import make_functional
from opake.optimizers import sgd
from opake.random import key

# Synthetic data
torch.manual_seed(42)
n_samples, n_features = 1000, 10
X = torch.randn(n_samples, n_features)
y = X @ torch.randn(n_features) + torch.randn(n_samples)

# Convert model to functional form
model = nn.Linear(n_features, 1)
fmodel, params = make_functional(model)


# Per-example loss (takes a single example, returns a scalar)
def loss_fn(params, example):
    """Return the squared error for one training example."""
    x, y_true = example
    y_pred = fmodel(params, x.unsqueeze(0)).squeeze()
    return (y_pred - y_true) ** 2


# Privacy parameters
epsilon, delta = 3.0, 1e-5
batch_size = 32
sample_rate = batch_size / n_samples
steps_per_epoch = n_samples // batch_size
num_steps = 10 * steps_per_epoch

# Calibrate noise multiplier for target epsilon
result = acc.calibrate(
    budget=acc.epsilon_budget(epsilon, delta=delta),
    process=lambda nm: (
        dpsgd_acc.poisson(dpsgd_acc.gaussian(nm), sample_rate) * num_steps
    ),
    param_min=0.1,
    param_max=100.0,
)
noise_multiplier = result.param

# Create DP components
grad_fn, clip_state = clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=1,
    clipping_norm=1.0,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(
    noise_multiplier=noise_multiplier,
    key=key(42),
)
dataset = TensorDataset(X, y)
sampler = PoissonSampler(
    dataset,
    sample_rate=sample_rate,
    n_steps=num_steps,
    key=key(43),
)
loader = DataLoader(dataset, batch_sampler=sampler)

# Privacy tracker
step_proc = dpsgd_acc.poisson(dpsgd_acc.gaussian(noise_multiplier), sample_rate)
accountant = Accountant(budget=acc.epsilon_budget(epsilon, delta=delta))

# Training loop
optimizer = sgd(lr=0.01)
opt_state = optimizer.init(params)

for step, batch in enumerate(loader, start=1):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = torchopt.apply_updates(params, updates)

    accountant = accountant | step_proc
    if step % steps_per_epoch == 0:
        epoch = step // steps_per_epoch
        print(f"Epoch {epoch}/10 - epsilon={accountant.epsilon_at(delta):.2f}")

final_epsilon = accountant.epsilon_at(delta)
accounting_tolerance = 1e-9
assert step == num_steps
assert abs(final_epsilon - result.achieved) < accounting_tolerance
