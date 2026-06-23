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

## References

- J. J. More, B. S. Garbow, K. E. Hillstrom. *Testing Unconstrained Optimization Software*. ACM TOMS, 7(1):17-41, 1981. [DOI:10.1145/355934.355936](https://doi.org/10.1145/355934.355936)
- E. G. Birgin, J. L. Gardenghi, J. M. Martinez, S. A. Santos, Ph. L. Toint. *Worst-case evaluation complexity for unconstrained nonlinear optimization using high-order regularized models*. Math. Program., 163:359-368, 2017. [DOI:10.1007/s10107-016-1065-8](https://doi.org/10.1007/s10107-016-1065-8)
- E. G. Birgin, J. L. Gardenghi, J. M. Martinez, S. A. Santos. *On the use of third-order models with fourth-order regularization for unconstrained optimization*. Optim. Lett., 14:815-838, 2020. [DOI:10.1007/s11590-019-01395-z](https://doi.org/10.1007/s11590-019-01395-z)
- C. Cartis, N. I. M. Gould, Ph. L. Toint. *Adaptive cubic regularisation methods for unconstrained optimization. Part I*. Math. Program., 127:245-295, 2011. [DOI:10.1007/s10107-009-0286-5](https://doi.org/10.1007/s10107-009-0286-5)

## License

MIT License. See [LICENSE](LICENSE).
