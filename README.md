# maxwellgp

A Gaussian process (GP) framework for modeling 3D time-harmonic electromagnetic
fields constrained by Maxwell's equations, built on JAX. The prior is an
Ehrenpreis-Palamodov plane-wave feature map, so every sample and the posterior
mean satisfy the homogeneous Maxwell system exactly; only the data (boundary or
interior field observations) are fit.

This is the general-purpose solver core. The cavity-specific physics and the
reaction-operator assembly live in
[cavity-epgp](https://github.com/luiswirth/cavity-epgp), which depends on this
package.

## Public API

- `MaxwellKernel(n_spectral, wavenumber, trace=..., key=...)` plane-wave feature
  map; `trace="tangential"` conditions on the tangential trace.
- `GaussianProcess(kernel, log_noise=...)` the regression core: marginal
  likelihood (`nlml`) and `condition(X, Y)`.
- `GaussianProcessPosterior` posterior `mean`, `cov`, `var` given feature
  evaluations.

A field point is a 3-vector; conditioning points carry a normal (a 6-vector,
position and normal stacked). The model predicts the 6-vector field `[E, B]`.

## Requirements

- [uv](https://docs.astral.sh/uv/)

## Example

```bash
uv run python examples/basic.py
```

`examples/basic.py` fits a known plane-wave superposition from sampled points and
reports the recovered-field RMSE, exercising the kernel, marginal-likelihood
training, and posterior mean.

## Development

```bash
uv run ruff check
uv run pyright
```
