"""Centering, PCA whitening and FastICA implemented from scratch with NumPy.

Shapes used everywhere: signals are (n_channels, n_samples).
"""
import numpy as np


def center(X):
    """Subtract each channel's mean. Returns (X_centered, mean)."""
    mean = X.mean(axis=1, keepdims=True)
    return X - mean, mean


def pca_whiten(X, n_components=None, eps=1e-12):
    """PCA whitening using the covariance matrix and its eigendecomposition.

    After whitening, Z = V @ X has identity covariance: the channels are
    uncorrelated and have unit variance.

    Returns Z (n_components, n_samples), V (whitening matrix), eigvals, eigvecs.
    """
    n = X.shape[1]
    C = (X @ X.T) / n                        # covariance (X is already centered)
    eigvals, eigvecs = np.linalg.eigh(C)     # symmetric -> real eigenvalues, ascending
    order = np.argsort(eigvals)[::-1]        # largest variance first
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    if n_components is not None:             # more mixtures than sources -> keep top k
        eigvals, eigvecs = eigvals[:n_components], eigvecs[:, :n_components]
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals + eps))
    V = D_inv_sqrt @ eigvecs.T               # V = D^(-1/2) E^T
    return V @ X, V, eigvals, eigvecs


# Non-linearities g (derivative of the contrast function G) and their derivatives g'
def _logcosh(u, a=1.0):
    t = np.tanh(a * u)
    return t, a * (1 - t ** 2)


def _exp(u):
    e = np.exp(-u ** 2 / 2)
    return u * e, (1 - u ** 2) * e


def _cube(u):
    return u ** 3, 3 * u ** 2


NONLINEARITIES = {"logcosh": _logcosh, "exp": _exp, "cube": _cube}


def _sym_decorrelate(W):
    """W <- (W W^T)^(-1/2) W  keeps the rows orthonormal (symmetric orthogonalisation)."""
    s, U = np.linalg.eigh(W @ W.T)
    return U @ np.diag(1.0 / np.sqrt(s)) @ U.T @ W


def fastica(Z, n_components=None, fun="logcosh", max_iter=500, tol=1e-6, seed=0):
    """Symmetric FastICA (Hyvarinen & Oja, 2000) on WHITENED data Z.

    Fixed-point update for every row w of W at once:
        w+ = E[z g(w^T z)] - E[g'(w^T z)] w
    followed by symmetric decorrelation so the components stay orthogonal.

    Returns W (unmixing matrix in whitened space), S = W @ Z, n_iterations, history.
    """
    k, n = Z.shape
    k = n_components or k
    g = NONLINEARITIES[fun]
    rng = np.random.default_rng(seed)
    W = _sym_decorrelate(rng.standard_normal((k, Z.shape[0])))
    history = []
    for it in range(1, max_iter + 1):
        U = W @ Z                                  # current estimates
        gU, gpU = g(U)
        W_new = (gU @ Z.T) / n - np.diag(gpU.mean(axis=1)) @ W
        W_new = _sym_decorrelate(W_new)
        # convergence: |<w_new, w_old>| -> 1 for every component (sign does not matter)
        delta = np.max(np.abs(np.abs(np.einsum("ij,ij->i", W_new, W)) - 1))
        history.append(delta)
        W = W_new
        if delta < tol:
            break
    return W, W @ Z, it, history


def ica(X, n_components=None, **kw):
    """Full pipeline: center -> whiten -> FastICA. Returns dict with everything."""
    Xc, mean = center(X)
    Z, V, eigvals, eigvecs = pca_whiten(Xc, n_components)
    W, S, n_iter, hist = fastica(Z, **kw)
    return {"S": S, "W": W, "V": V, "Z": Z, "mean": mean, "unmixing": W @ V,
            "eigvals": eigvals, "n_iter": n_iter, "history": hist}
