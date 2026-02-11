import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


N = 5000
fs = 48000
Ts = 1 / fs
t = np.arange(N) * Ts

L = 8
mu = 0.01
x = np.random.randn(N)

# Adaptive filter initialization
w = np.zeros(L)

# Primary path
p = np.array([0.5, -0.3, 0.2, 0.1, 1, -1, 0.3, 0.5])

# History
e_hist = np.zeros(N)
w_hist = np.zeros((N, L))

for n in range(L, N):
    # Entry vector (a slice of x)
    x_vec = x[n : n - L : -1]

    # Desired signal
    d = np.dot(p, x_vec)

    # Filter output
    y = np.dot(w, x_vec)

    # Error signal
    e = d - y
    e_hist[n] = e

    # LMS algorithm
    w = w + 2*mu*x_vec*e
    w_hist[n, :] = w

plt.figure()
plt.stem(p, linefmt='C0-', markerfmt='C0o', basefmt=" ")
plt.stem(w, linefmt='C1-', markerfmt='C1s', basefmt=" ")
plt.legend(['Primary path p', 'Adaptive filter w'])
plt.title('Final coefficients')
plt.show()

plt.figure()
plt.plot(t, 10*np.log10(e_hist**2 + 1e-12))
plt.title('Error power')
plt.xlabel('Time [s]')
plt.ylabel('Power [dB]')
plt.show()

plt.figure()
for k in range(L):
    plt.plot(t, w_hist[:, k])
plt.title('Coefficient convergence')
plt.xlabel('Time [s]')
plt.ylabel('Value')
plt.show()




    





