import numpy as np
import function_helper as fh


def armLineSearch_clean(func, x0, sigma0, alpha=1/2, eps=1e-8, max_iter=1e5):
    k = 0
    xk = np.copy(x0)
    sigmak = sigma0
    coef = 1.0
    slope = 1e-2
    fxk, df = func.evaluate(xk, order=1)

    while k < max_iter:
        if np.linalg.norm(df, ord=np.inf) < eps:
            break
        l = 0
        while True:
            sk = -coef * sigmak * df
            xkl = xk + sk
            f_trial = func.evaluate(xkl, order=0)
            if np.linalg.norm(sk) < 1e-15 * (1 + np.linalg.norm(xk)):
                break
            if fxk - f_trial >= -df @ sk * slope:
                break
            l += 1
            coef *= alpha
        xk = xk + sk
        coef = 1.0
        sigmak = sigmak * alpha**(l - 1)
        fxk, df = func.evaluate(xk, order=1)
        k += 1

    return xk


def alsBfgsUpdate_clean(func, x0, sigma0, alpha=1/2, eps=1e-8, max_iter=1e5,
                        subprodata=None):
    k = 0
    xk = np.copy(x0)
    Id = np.eye(func.n)
    sigmak = sigma0
    Hk = np.copy(Id)
    coef = 1.0
    fxk, df = func.evaluate(xk, order=1)
    slope = 1e-2
    extended = max_iter >= 1e3
    outer_converged = False

    while k < max_iter:
        if subprodata is None:
            cond = np.linalg.norm(df, ord=np.inf) < eps
        else:
            max_iter = np.min([max_iter, 1e3])
            if subprodata[0] == "CRNM":
                cond = np.linalg.norm(df) <= subprodata[1] * np.linalg.norm(xk - x0)**2 / 2
            elif subprodata[0] == "ATO":
                cond = np.linalg.norm(df) <= subprodata[1] * np.linalg.norm(xk - x0)**3 / 2
        if cond:
            break

        if k == max_iter - 1 and not extended and subprodata is not None and len(subprodata) >= 3:
            _, dfx_outer = subprodata[2].evaluate(xk, order=1)
            if np.isfinite(np.linalg.norm(dfx_outer, ord=np.inf)) and np.linalg.norm(dfx_outer, ord=np.inf) < eps:
                outer_converged = True
                break
            else:
                max_iter = 1000
                extended = True

        l = 0
        while True:
            sk = -coef * sigmak * Hk @ df
            xkl = xk + sk
            f_trial = func.evaluate(xkl, order=0)
            if np.linalg.norm(sk) < 1e-15 * (1 + np.linalg.norm(xk)):
                break
            if fxk - f_trial >= -df @ sk * slope:
                break
            l += 1
            coef *= alpha

        xkp1 = xk + sk
        coef = 1.0
        sigmak = sigmak * alpha**(l - 1)
        fxk, dfkp1 = func.evaluate(xkp1, order=1)
        yk = dfkp1 - df
        denom = np.dot(sk, yk)
        if denom > 0.0:
            numer = np.outer(sk, yk)
            Hk = (Id - numer/denom) @ Hk @ (Id - numer.T/denom) + np.outer(sk, sk)/denom
        xk = np.copy(xkp1)
        df = np.copy(dfkp1)
        k += 1

    return xk, outer_converged, k


def cubRegNewtonMethod_clean(func, x0, sigma0, alpha=1/2, eps=1e-8, max_iter=1e3,
                              iter_update=False, use_theta=False):
    k = 0
    eta1 = .01
    eta2 = .95
    gamma = 3.0
    xk = np.copy(x0)
    sigmamin = 1e-12
    sigmak = np.copy(sigma0)
    thetak = sigmak
    fxk, dfxk, ddfxk = func.evaluate(xk, order=2)
    inner_iters = 1000
    alpha_iters = 100
    max_inner_k_accepted = 0

    while k < max_iter:
        if np.linalg.norm(dfxk, ord=np.inf) < eps:
            return xk

        thetak = max(thetak, sigmak)
        subpro_sigma = thetak if use_theta else sigmak
        curr_M = fh.M_x_sig(sigmak, xk, fxk, dfxk, ddfxk)
        xpred, outer_conv, inner_k = alsBfgsUpdate_clean(
            curr_M, xk, 1.0, max_iter=int(inner_iters),
            subprodata=['CRNM', subpro_sigma, func] if iter_update else ['CRNM', subpro_sigma],
            eps=eps)

        if outer_conv:
            return xpred

        if np.linalg.norm(xk - xpred) < 1e-16:
            return xk

        d = xpred - xk
        delta_pred = -(dfxk @ d + 0.5 * d @ ddfxk @ d + sigmak / 2 * np.linalg.norm(d)**3)
        f_pred = func.evaluate(xpred, order=0)
        delta_act = fxk - f_pred

        if delta_pred <= 0:
            return xk

        rho_k = delta_act / delta_pred
        if rho_k >= eta1:
            xk = np.copy(xpred)
            fxk, dfxk, ddfxk = func.evaluate(xk, order=2)
            if rho_k >= eta2:
                sigmak = np.max([sigmamin, sigmak * alpha])
            max_inner_k_accepted = max(max_inner_k_accepted, inner_k)
            alpha_iters /= 1.5
            if iter_update:
                inner_iters = max_inner_k_accepted * (1.5 + alpha_iters)
        else:
            sigmak = sigmak * gamma
        k += 1

    return xk


def adaptiveThirdOrder_clean(func, x0, sigma0, alpha=1/2, eps=1e-8, max_iter=1e3,
                              iter_update=False, use_theta=False):
    k = 0
    eta1 = .01
    eta2 = .95
    gamma = 3.0
    xk = np.copy(x0)
    sigmamin = 1e-12
    sigmak = np.copy(sigma0)
    thetak = sigmak
    fxk, dfxk, ddfxk, dddfxk = func.evaluate(xk, order=3)
    inner_iters = 1000
    alpha_iters = 100
    max_inner_k_accepted = 0

    while k < max_iter:
        if np.linalg.norm(dfxk, ord=np.inf) < eps:
            return xk

        thetak = max(thetak, sigmak)
        subpro_sigma = thetak if use_theta else sigmak
        curr_O = fh.Omega_x_sig(sigmak, xk, fxk, dfxk, ddfxk, dddfxk)
        xpred, outer_conv, inner_k = alsBfgsUpdate_clean(
            curr_O, xk, 1.0, max_iter=int(inner_iters),
            subprodata=['ATO', subpro_sigma, func] if iter_update else ['ATO', subpro_sigma],
            eps=eps)

        if outer_conv:
            return xpred

        if np.linalg.norm(xk - xpred) < 1e-16:
            return xk

        d = xpred - xk
        delta_pred = -(dfxk @ d + 0.5 * d @ ddfxk @ d
                       + (1/6) * np.einsum('ijk,i,j,k->', dddfxk, d, d, d)
                       + sigmak / 8 * np.linalg.norm(d)**4)
        f_pred = func.evaluate(xpred, order=0)
        delta_act = fxk - f_pred

        if delta_pred <= 0:
            return xk

        rho_k = delta_act / delta_pred
        if rho_k >= eta1:
            xk = np.copy(xpred)
            fxk, dfxk, ddfxk, dddfxk = func.evaluate(xk, order=3)
            if rho_k >= eta2:
                sigmak = np.max([sigmamin, sigmak * alpha])
            max_inner_k_accepted = max(max_inner_k_accepted, inner_k)
            alpha_iters /= 1.5
            if iter_update:
                inner_iters = max_inner_k_accepted * (1.5 + alpha_iters)
        else:
            sigmak = sigmak * gamma
        k += 1

    return xk
