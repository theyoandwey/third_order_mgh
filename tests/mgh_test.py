"""Derivative verification and Taylor-model visualisation for MGH functions."""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import MGH
import function_helper as fh
from MGH import *


def relative_error(numeric, analytic, eps=1e-12):
    denom = max(np.linalg.norm(analytic), np.linalg.norm(numeric), eps)
    return np.linalg.norm(numeric - analytic) / denom


def test_all_derivatives(func, x, h=1e-5, tol=1e-6, abs_tol=1e-10):
    x = np.asarray(x, dtype=float)
    n = len(x)

    def check(err, numeric, analytic):
        both_small = max(np.linalg.norm(analytic), np.linalg.norm(numeric)) < 1e-6
        if both_small:
            return np.linalg.norm(numeric - analytic) < 1e-7
        return err < tol

    f0, grad_analytic, hess_analytic, T_analytic = func.evaluate(x, order=3)

    # Gradient check: finite-diff on f
    grad_numeric = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        fp = func.evaluate(x + h*e, order=0)
        fm = func.evaluate(x - h*e, order=0)
        grad_numeric[i] = (fp - fm) / (2*h)

    grad_err = relative_error(grad_numeric, grad_analytic)
    grad_ok = check(grad_err, grad_numeric, grad_analytic)

    # Hessian check: finite-diff on grad
    hess_numeric = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = 1.0
        _, gp = func.evaluate(x + h*e, order=1)
        _, gm = func.evaluate(x - h*e, order=1)
        hess_numeric[i, :] = (gp - gm) / (2*h)

    hess_err = relative_error(hess_numeric, hess_analytic)
    hess_ok = check(hess_err, hess_numeric, hess_analytic)

    # Tensor check: finite-diff on hess
    def make_tensor_symmetric(T):
        n = T.shape[0]
        T_sym = np.zeros((n, n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    idx = sorted([i, j, k], reverse=True)
                    T_sym[i, j, k] = T[idx[0], idx[1], idx[2]]
        return T_sym

    T_analytic = make_tensor_symmetric(T_analytic)
    T_numeric = np.zeros((n, n, n))
    for k in range(n):
        e = np.zeros(n); e[k] = 1.0
        _, _, hp = func.evaluate(x + h*e, order=2)
        _, _, hm = func.evaluate(x - h*e, order=2)
        T_numeric[:, :, k] = (hp - hm) / (2*h)

    tensor_err = relative_error(T_numeric, T_analytic)
    tensor_ok = check(tensor_err, T_numeric, T_analytic)

    print("\n===== GRADIENT CHECK =====")
    print("Analytical:", grad_analytic)
    print("Numeric:   ", grad_numeric)
    print("Relative error:", grad_err)
    print("Result:", "OK" if grad_ok else "BAD")

    print("\n===== HESSIAN CHECK =====")
    print("Analytical:\n", hess_analytic)
    print("Numeric:\n", hess_numeric)
    print("Relative error:", hess_err)
    print("Result:", "OK" if hess_ok else "BAD")

    print("\n===== THIRD-ORDER TENSOR CHECK =====")
    print("Analytical:\n", T_analytic)
    print("Numeric:\n", T_numeric)
    print("Relative error:", tensor_err)
    print("Result:", "OK" if tensor_ok else "BAD")

    return {
        "grad_err": grad_err, "hess_err": hess_err, "tensor_err": tensor_err,
        "grad_ok": grad_ok, "hess_ok": hess_ok, "tensor_ok": tensor_ok,
        "grad_numeric": grad_numeric, "hess_numeric": hess_numeric, "tensor_numeric": T_numeric,
    }


def plot_taylor_models(func, x0, delta=2.0, n=100, sig=1.0):
    x0 = np.atleast_1d(x0)
    ndim = func.n
    delta = np.atleast_1d(delta)
    if len(delta) == 1:
        delta = np.full(ndim, delta[0])

    fx = func.f(x0)
    gradx = func.grad(x0)
    hessx = func.hess(x0)

    if ndim == 2:
        thirdx = func.third_order_tensor(x0)
        M = fh.M_x_sig(sig, x0, fx, gradx, hessx)
        O = fh.Omega_x_sig(sig, x0, fx, gradx, hessx, thirdx)

        g1 = np.linspace(x0[0] - delta[0], x0[0] + delta[0], n)
        g2 = np.linspace(x0[1] - delta[1], x0[1] + delta[1], n)

        Z_f = np.array([[float(func.f([i, j])) for j in g2] for i in g1])
        Z_m = np.array([[float(M.f([i, j])) for j in g2] for i in g1])
        Z_o = np.array([[float(O.f([i, j])) for j in g2] for i in g1])

        Z_f = np.log10(np.abs(Z_f) + 1e-30)
        Z_m = np.log10(np.abs(Z_m) + 1e-30)
        Z_o = np.log10(np.abs(Z_o) + 1e-30)

        vmin = min(Z_f.min(), Z_m.min(), Z_o.min())
        vmax = max(Z_f.max(), Z_m.max(), Z_o.max())
        levels = np.linspace(vmin, vmax, 50)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, Z, title in zip(axes, [Z_f, Z_m, Z_o],
                                ["f", "M (order 2)", "Ω (order 3)"]):
            im = ax.contourf(g2, g1, Z, levels=levels, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.plot(x0[1], x0[0], 'r*', markersize=15, zorder=5)
            ax.set_xlabel("x2"); ax.set_ylabel("x1")
            ax.set_title(f"{title} (log10)")
            plt.colorbar(im, ax=ax)

    else:
        thirdx = func.third_order_tensor(x0)
        M = fh.M_x_sig(sig, x0, fx, gradx, hessx)
        O = fh.Omega_x_sig(sig, x0, fx, gradx, hessx, thirdx)

        fig, axes = plt.subplots(1, ndim, figsize=(5 * ndim, 4))
        if ndim == 1:
            axes = [axes]

        for d in range(ndim):
            t = np.linspace(-delta[d], delta[d], n)
            for label, model, color in [("f", func, '#1a5276'),
                                        ("M", M, '#922b21'),
                                        ("Ω", O, '#1d6a3a')]:
                vals = []
                for ti in t:
                    xp = np.copy(x0)
                    xp[d] += ti
                    vals.append(model.f(xp))
                axes[d].plot(t, vals, label=label, color=color, linewidth=1.5)
            axes[d].axvline(0, color='gray', linestyle='--', alpha=0.5)
            axes[d].set_xlabel(f"x{d+1} perturbation")
            axes[d].set_ylabel("f")
            axes[d].set_title(f"Slice along x{d+1}")
            axes[d].legend(fontsize=8)
            axes[d].grid(True, alpha=0.3)

    plt.suptitle(f"Taylor models at x0 = {np.round(x0, 3)}, σ = {sig}", fontsize=12)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    np.random.seed(0)

    # Full derivative check across all MGH functions (uncomment to run)
    # test_functions = [
    #     Rosenbrock, FreudensteinAndRoth, PowellBadlyScaled, BrownBadlyScaled,
    #     Beale, JennrichAndSampson, HelicalValley, Bard, Gaussian, Meyer,
    #     GulfResearchAndDevelopement, Box3D, PowellSingular, Wood,
    #     KowalikAndOsborne, BrownAndDennis, Osborne1, BiggsEXP6, Osborne2,
    #     Watson, ExtendedRosenbrock, ExtendedPowellSingular, PenaltyI,
    #     PenaltyII, VariablyDimensioned, Trigonometric, BrownAlmostLinear,
    #     DiscreteBoundaryValue, DiscreteIntegralEquation, BroydenTridiagonal,
    #     BroydenBanded, LinearFullRank, LinearRank1,
    #     LinearRank1ZeroColumnsAndRows, Chebyquad
    # ]
    # results = []
    # for cls in test_functions:
    #     func = cls()
    #     x_test = func.x0 + 0.1 * np.random.randn(func.n)
    #     x_test = np.where(np.abs(x_test) < 1e-3, 0.1, x_test)
    #     print(f"\n{'='*60}")
    #     print(f"Testing {cls.__name__} (n={func.n}, m={func.m})")
    #     r = test_all_derivatives(func, x_test, h=1e-5)
    #     r['name'] = cls.__name__
    #     results.append(r)
    # print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    # print(f"{'Function':<30} {'Grad':>8} {'Hess':>8} {'Tensor':>8}")
    # print("-" * 56)
    # for r in results:
    #     g = "OK" if r['grad_ok'] else f"FAIL({r['grad_err']:.1e})"
    #     h = "OK" if r['hess_ok'] else f"FAIL({r['hess_err']:.1e})"
    #     t = "OK" if r['tensor_ok'] else f"FAIL({r['tensor_err']:.1e})"
    #     print(f"{r['name']:<30} {g:>8} {h:>8} {t:>8}")

    # Quick check on HelicalValley and Bard
    for cls, x_test in [
        (HelicalValley, np.array([-1.2, 0.3, 0.5])),
        (Bard,          np.array([0.8, 1.2, 0.9])),
    ]:
        func = cls()
        print(f"\n{'='*60}")
        print(f"Testing {cls.__name__} (n={func.n}, m={func.m})")
        f_val    = func.f(x_test)
        f_manual = sum(func.fi(x_test, i+1)**2 for i in range(func.m))
        print(f"f(x)        = {f_val:.10f}")
        print(f"sum fi(x)^2 = {f_manual:.10f}")
        print(f"Match: {'OK' if abs(f_val - f_manual) < 1e-12 else 'BAD'}")
        test_all_derivatives(func, x_test, h=1e-5)
