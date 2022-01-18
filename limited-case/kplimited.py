# With FFT(Fast Fourier Transform method)
# KP limited case
# U_t + U * U_x = nu * U_xx when nu = 0 - inviscid Burgers' equation

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
import matplotlib.cm as cm
plt.rcParams['figure.figsize'] = [12, 12]
plt.rcParams.update({'font.size': 18})

nu = 0       # Diffusion constant
L = 20       # Length of domain
N = 1000     # Number of discretization points
dx = L/N
x = np.arange(-L/2, L/2, dx) # Define x domain

# Define discrete wave numbers
kappa = 2*np.pi*np.fft.fftfreq(N, d=dx)

# Initial conditions
u0 = 1 / np.cosh(x)

# Simulate PDE in spatial domain
dt = 0.025
t = np.arange(0, 800*dt, dt)

def rhsBurgers(u, t, kappa, nu):
    uhat = np.fft.fft(u)
    d_uhat = (1j)*kappa*uhat
    dd_uhat = -np.power(kappa, 2) * uhat
    d_u = np.fft.ifft(d_uhat)
    dd_u = np.fft.ifft(dd_uhat)
    du_dt = - u * d_u + nu * dd_u
    return du_dt.real

u = odeint(rhsBurgers, u0, t, args=(kappa, nu))

# Waterfall plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# u_plot = u[0:-1:10, :]
# for j in range(u_plot.shape[0]):
#     ys = j*np.ones(u_plot.shape[1])
#     ax.plot(x, ys, u_plot[j, :], color=cm.jet(j*20))
#
# plt.figure()
plt.imshow(np.flipud(u), aspect = 8)
plt.title(label='KP limited case:',
          loc="left",
          color='blue',
          fontsize=10,
          )
plt.title(label="© George Meshveliani",
          loc="right",
          fontsize=6,)
plt.xlabel('x')
plt.ylabel('t')
# plt.axis('off')

plt.set_cmap('jet')
plt.show()