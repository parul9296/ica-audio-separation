"""Evaluation that accounts for ICA's scale and permutation ambiguity."""
import numpy as np
from scipy.optimize import linear_sum_assignment  # Hungarian matching only (not ICA)


def _std(x):
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def corr_matrix(S_true, S_est):
    """|Pearson correlation| between every true source (rows) and every estimate (cols)."""
    A = np.array([_std(s) for s in S_true])
    B = np.array([_std(s) for s in S_est])
    return np.abs(A @ B.T)


def match(S_true, S_est):
    """Solve the permutation ambiguity: best one-to-one pairing by |correlation|."""
    C = corr_matrix(S_true, S_est)
    rows, cols = linear_sum_assignment(-C)
    return cols[np.argsort(rows)], C          # perm[i] = estimate matched to source i


def si_sdr(ref, est):
    """Scale-invariant SDR in dB (Le Roux et al., 2019). Fixes scale AND sign ambiguity."""
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = (est @ ref) / (ref @ ref + 1e-12)   # optimal scaling (negative alpha fixes sign)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((target @ target) / (noise @ noise + 1e-12))


def align(S_true, S_est):
    """Reorder, flip sign and rescale estimates so they line up with the true sources."""
    perm, C = match(S_true, S_est)
    out = []
    for i, j in enumerate(perm):
        e = S_est[j] - S_est[j].mean()
        r = S_true[i] - S_true[i].mean()
        out.append(e * (e @ r) / (e @ e + 1e-12))
    return np.array(out), perm, C


def evaluate(S_true, S_est):
    """Per-source |corr| and SI-SDR after optimal matching."""
    perm, C = match(S_true, S_est)
    return [{"corr": float(C[i, j]), "si_sdr": float(si_sdr(S_true[i], S_est[j]))}
            for i, j in enumerate(perm)], perm


def kurtosis(x):
    """Excess kurtosis: 0 for a Gaussian, >0 for peaky (super-Gaussian) signals like speech."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    return float((x ** 4).mean() - 3)
