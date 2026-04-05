from warnings import warn
import numpy as np
from scipy.fft import fft, ifft
from numpy.linalg import cholesky, solve
from scipy.signal import lfilter
import matplotlib.pyplot as plt

""" This code was taken from https://github.com/neil-gallagher/directed-spectrum 
    and modified lightly to fit the stochastic multichannel ANC algorithm
    
Pseudocode
Input:
    S(f)  ← matriz espectral (CPSD)
    max_iter, tol

# --- Inicialización ---
R(τ) = ifft(S(f))
R0   = R(0)

h    = chol(R0)          # raíz inicial
ψ(f) = h                 # constante en frecuencia

# --- Iteración ---
for iter = 1 ... max_iter:

    # 1. Evaluar error (whitening)
    E(f) = ψ(f)^(-1) · S(f) · ψ(f)^(-H)

    # 2. Construir corrección
    G(f) = E(f) + I

    # 3. Proyección causal
    G_plus(f) = causal_part( G(f) )

    # 4. Update
    ψ(f) = ψ(f) · G_plus(f)

    # 5. Check convergencia
    if ||ψ_new - ψ_old|| < tol:
        break

# --- Resultado ---
ψ(f) ≈ factor espectral tal que:
S(f) ≈ ψ(f) ψ^H(f)

# (opcional)
H(f), Σ ← reparametrización de ψ
"""

def _wilson_factorize(cpsd, max_iter, tol, eps_multiplier=100):
    """Factorize CPSD into transfer matrix (H) and covariance (Sigma).

    Implements the algorithm outlined in the following reference:
    G. Tunnicliffe. Wilson, “The Factorization of Matricial Spectral
    Densities,” SIAM J. Appl. Math., vol. 23, no. 4, pp. 420426, Dec.
    1972, doi: 10.1137/0123044.

    This code is based on an original implementation in MATLAB provided
    by M. Dhamala (mdhamala@mail.phy-ast.gsu.edu).

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix.
    f_samp : float
        Sampling rate of time series data X.
    max_iter : int
        Max number of Wilson factorization iterations.
    tol : float
        Wilson factorization convergence tolerance value.
    eps_multiplier : int
        Constant multiplier used in stabilizing the Cholesky decomposition
        for positive semidefinite CPSD matrices.

    Returns
    -------
    H : numpy.ndarray
        shape (n_windows, n_frequencies, n_signals, n_signals)
        Wilson factorization solutions for transfer matrix.
    Sigma : numpy.ndarray
        shape (n_windows, n_signals, n_signals)
        Wilson factorization solutions for innovation covariance matrix.
    """
    cpsd_cond = np.linalg.cond(cpsd) # calculates de spread of singular values
    if np.any(cpsd_cond > (1/ np.finfo(cpsd.dtype).eps)): 
        warn('CPSD matrix is singular within numerical tolerance, which may produce inaccurate results.')
        # Add diagonal of small values to cross-power spectral matrix to prevent
        # it from being negative semidefinite due to rounding errors
        this_eps = np.spacing(np.abs(cpsd)).max()
        cpsd = cpsd + np.eye(cpsd.shape[-1])*this_eps*eps_multiplier

    # Intial estimation
    psi, A0 = _init_psi(cpsd)

    L = cholesky(cpsd)
    #eigval, eigvec = eigh(cpsd)
    #eigval[eigval<0] = 0
    #L = np.sqrt(eigval[...,np.newaxis,:]) * eigvec

    H = np.zeros_like(psi)
    Sigma = np.zeros_like(A0)

    for w in range(cpsd.shape[0]): # window iteration wouldnt be needed
        for i in range(max_iter):
            # These lines implement: g = psi \ cpsd / psi* + I
            psi_inv_cpsd = solve(psi[w], L[w])
            g = psi_inv_cpsd @ psi_inv_cpsd.conj().transpose(0, 2, 1)
            g = g + np.identity(cpsd.shape[-1])

            gplus, g0 = _plus_operator(g)

            # S is chosen so that g0 + S is upper triangular; S + S* = 0
            S = -np.tril(g0, -1)
            S = S - S.conj().transpose()
            gplus = gplus + S
            psi_prev = psi[w].copy()
            psi[w] = psi[w] @ gplus

            A0_prev = A0[w].copy()
            A0[w] = A0[w] @ (g0 + S)

            if (_check_convergence(psi[w], psi_prev, tol) and
                    _check_convergence(A0[w], A0_prev, tol)):
                break
        else:
            warn('Wilson factorization failed to converge.', stacklevel=2)

        # right-side solve
        H[w] = (solve(A0[w].T, psi[w].transpose(0, 2, 1))).transpose(0, 2, 1)
        Sigma[w] = (A0[w] @ A0[w].T)
    return (H, Sigma)

##############################################################################################################################

def _init_psi(cpsd):
    """Return initial psi value for wilson factorization.

    Parameters
    ----------
    cpsd : numpy.ndarray
        Cross power spectral density matrix.

    Returns
    -------
    psi : numpy.ndarray
        shape (n_windows, n_frequencies, n_groups, n_groups)
        Initial value for psi used in Wilson factorization.
    h : numpy.ndarray
        shape (n_windows, n_groups, n_groups)
        Initial value for A0 used in Wilson factorization.
    """
    # TODO: provide other initialization options; test which is best.
    gamma = ifft(cpsd, axis=1)

    gamma0 = gamma[:, 0]

    # remove assymetry in gamma0 due to rounding error.
    gamma0 = np.real((gamma0 + gamma0.conj().transpose(0, 2, 1)) / 2.0)
    h = cholesky(gamma0).conj().transpose(0, 2, 1)
    psi = np.tile(h[:, np.newaxis], (1, cpsd.shape[1], 1, 1)).astype(complex)
    return psi, h

##############################################################################################################################

def _plus_operator(g):
    """Remove all negative lag components from time-domain representation.

    Parameters
    ----------
    g: numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Frequency-domain representation to which transformation will be applied.

    Returns
    -------
    g_pos : numpy.ndarray
        shape (n_frequencies, n_groups, n_groups)
        Transformed version of g with negative lag components removed.
    gamma[0] : numpy.ndarray
        shape (n_groups, n_groups)
        Zero-lag component of g in time-domain.
    """
    # remove imaginary components from ifft due to rounding error.
    gamma = ifft(g, axis=0).real

    # take half of 0 lag
    gamma[0] *= 0.5

    # take half of nyquist component if fft had even # of points
    F = gamma.shape[0]
    N = np.floor(F/2).astype(int)
    if F % 2 == 0:
        gamma[N] *= 0.5

    # zero out negative frequencies
    gamma[N+1:] = 0

    gp = fft(gamma, axis=0)
    return gp, gamma[0]

##############################################################################################################################

def _check_convergence(x, x0, tol):
    """Determine whether maximum relative change is lower than tolerance.

    Parameters
    ----------
    x : numpy.dnarray
        Current matrix/array.
    x0 : numpy.ndarray
        Previous matrix/array
    tol : float
        Tolerance value for convergence check.

    Returns
    -------
    converged : bool
        True indicates convergence has occured, False indicates otherwise.
    """
    x_diff = np.abs(x - x0)
    ab_x = np.abs(x)
    this_eps = np.finfo(ab_x.dtype).eps
    ab_x[ab_x <= 2*this_eps] = 1
    rel_diff = x_diff / ab_x
    converged = rel_diff.max() < tol
    return converged

# Shapes:
# X  : (K, n_freqs)          → señales multicanal en frecuencia
# F  : (n_freqs, K, K)       → matriz por frecuencia

# Para cada frecuencia f:
# x_f = X[:, f]              → (K,)     vector de canales
# F_f = F[f]                 → (K, K)   mezcla entre canales

# Whitening:
# v_f = F_f^{-1} @ x_f       → (K,)     canales decorrelacionados

# Intuición:
# cada frecuencia tiene su propio "desmezclador" F(f)

if __name__ == "__main__":

    blocklength = 128
    nBlocks = 500
    N = nBlocks * blocklength
    v1 = np.random.randn(N)
    v2 = np.random.randn(N)
    x1 = lfilter([1], [1, -0.9], v1)
    x2 = lfilter([1], [1, -0.5], v2)
    x = np.stack([x1, x2], axis=0)
    alpha = 0.5   
    SxxAvg = None
    SvvAvg = None
    F = None

    for k in range(nBlocks):
        xSignals = x[:, k*blocklength:(k+1)*blocklength]
        X = np.fft.rfft(xSignals, axis=1)
        
        nChannels, nFreqs = X.shape
        Sxx = np.zeros((nFreqs, nChannels, nChannels), dtype=complex)

        for f in range(nFreqs):
            xf = X[:, f]
            Sxx[f] = xf[:, None] @ xf[None, :].conj()
        
        if SxxAvg is None:
            SxxAvg = Sxx.copy()
        else:
            SxxAvg = alpha*SxxAvg + (1-alpha)*Sxx
        
        if(k>19):
            cpsd = SxxAvg[np.newaxis, ...]
            H, Sigma = _wilson_factorize(cpsd, 100, 1e-6)
            F = H[0] @ np.linalg.cholesky(Sigma[0])
            
        if F is not None:
            Svv = np.zeros((nFreqs, nChannels, nChannels), dtype=complex)

            for f in range(nFreqs):
                Finv = np.linalg.inv(F[f])
                Svv[f] = Finv @ SxxAvg[f] @ Finv.conj().T

            if SvvAvg is None:
                SvvAvg = Svv.copy()
            else:
                SvvAvg = alpha*SvvAvg + (1-alpha)*Svv
    diag = np.real(np.diagonal(SvvAvg, axis1=1, axis2=2))
    off = np.abs(SvvAvg[:,0,1]) 

    plt.figure()
    plt.plot(10*np.log10(diag[:,0]), label="ch1")
    plt.plot(10*np.log10(diag[:,1]), label="ch2")
    plt.plot(10*np.log10(off), label="off-diag")
    plt.legend()
    plt.title("Whitened spectrum")
    plt.show()

    """
    # Offline testing
    blocklength = 128
    nBlocks = 500
    N = nBlocks * blocklength
    v1 = np.random.randn(N)
    v2 = np.random.randn(N)

    x1 = lfilter([1], [1, -0.9], v1)
    x2 = lfilter([1], [1, -0.5], v2)
    x = np.stack([x1, x2], axis=0)

    # señales completas
    X = np.fft.rfft(x, axis=1)   # (K, nFreqs)

    nChannels, nFreqs = X.shape
    Sxx = np.zeros((nFreqs, nChannels, nChannels), dtype=complex)

    for f in range(nFreqs):
        xf = X[:, f]
        Sxx[f] = xf[:, None] @ xf[None, :].conj()

    # Wilson
    cpsd = Sxx[np.newaxis, ...]
    H, Sigma = _wilson_factorize(cpsd, 100, 1e-6)
    F = H[0] @ np.linalg.cholesky(Sigma[0])

    V = np.zeros_like(X, dtype=complex)

    for f in range(nFreqs):
        V[:, f] = np.linalg.solve(F[f], X[:, f])
        
    Svv = np.zeros((nFreqs, nChannels, nChannels), dtype=complex)

    for f in range(nFreqs):
        vf = V[:, f]
        Svv[f] = vf[:, None] @ vf[None, :].conj()
        
    diag = np.real(np.diagonal(Svv, axis1=1, axis2=2))
    off  = np.abs(Svv[:,0,1])  # para 2 canales

    plt.figure()
    plt.plot(10*np.log10(diag[:,0]/np.mean(diag[:,0]) + 1e-12), label="ch1")
    plt.plot(10*np.log10(diag[:,1]/np.mean(diag[:,1]) + 1e-12), label="ch2")
    plt.plot(10*np.log10(off/np.mean(off) + 1e-12), label="off-diag")
    plt.legend()
    plt.title("Whitened spectrum")
    plt.show()
    """