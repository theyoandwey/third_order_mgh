# Third-Order Methods for Nonconvex Optimization

Implementation of p-th order regularized methods (orders 1 to 3) benchmarked on the More-Garbow-Hillstrom (MGH) test set.

## Files

- **`MGH.py`** — 35 MGH test functions with analytical gradient, Hessian, and third-order tensor
- **`function_helper.py`** — Regularized Taylor models: cubic model `M_x_sig` (order 2) and quartic model `Omega_x_sig` (order 3)
- **`methods_clean.py`** — Optimization methods: Armijo linesearch, ALS-BFGS, cubic regularized Newton, adaptive third-order

## Usage

```bash
pip install -r requirements.txt
```

```python
import numpy as np
from MGH import Rosenbrock
from methods_clean import cubRegNewtonMethod_clean, adaptiveThirdOrder_clean

func = Rosenbrock()
x, *_ = cubRegNewtonMethod_clean(func, func.x0, sigma0=1.0)
```

## Requirements

- Python 3.8+
- NumPy
