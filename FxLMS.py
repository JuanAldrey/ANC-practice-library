import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

N = 100000
fs = 1000
Ts = 1 / fs
t = np.arange(N) * Ts

L = 8
mu = 5e-4
x = np.random.randn(N)

# Adaptive filter initialization
w = np.zeros(L)

# Primary path
p = np.array([0.5, -0.3, 0.2, 0.1, 1, -1, 0.3, 0.5])
p_delayed = np.concatenate([np.zeros(8), p])

# Secondary path
s = np.array([0.1, -0.1, 0.5, 0.8, 0.7, 1, 0.4, 0.3])

# Filtered X output initialization
f_x_hist = np.zeros(N)

# Adaptive filter output initilization
y_hist = np.zeros(N)

# Desired signal output history
d_hist = np.zeros(N)

# History
e_hist = np.zeros(N)
w_hist = np.zeros((N, L))

for n in range(2*L, N):
    # Entry vector (a slice of x)
    x_vec = x[n : n - L : -1]

    # Desired signal
    d_hist[n] = np.dot(p, x_vec)
    d_vec = d_hist[n : n - L : -1]
    d = d_vec[0]

    # Filter output with secondary path
    y_hist[n] = np.dot(w, x_vec)
    y_vec = y_hist[n : n - L : -1]
    y_f = np.dot(s, y_vec)

    # Error signal
    e = d - y_f
    e_hist[n] = e

    # Filtered X
    f_x_hist[n] = np.dot(s, x_vec)
    f_x_vec = f_x_hist[n: n - L: -1]

    # LMS algorithm
    w = w + mu*f_x_vec*e
    w_hist[n, :] = w

plt.figure()
plt.stem(p, linefmt='C0-', markerfmt='C0o', basefmt=" ")
plt.stem(np.convolve(w, s)[:L], linefmt='C1-', markerfmt='C1s', basefmt=" ")
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




    





