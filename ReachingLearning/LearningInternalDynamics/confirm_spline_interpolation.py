import numpy as np
import casadi as cas
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt


# Initialize results plot
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
axs = [ax1, ax2]

# --- Test 1D version --- #
xpoints = np.array([1.1, 1.2, 1.3, 1.8,
                    2.3, 2.4, 2.5, 2.8,
                    3.1, 3.3, 3.8, 3.9,
                    4.2, 4.5, 4.7, 4.9,
                    5.1, 5.2, 5.5, 5.8])
ypoints = xpoints * np.sin(5*xpoints) + xpoints**1/3 + np.random.normal(0, 0.1, xpoints.shape)

# Create an interpolant in scipy
interp_func_2D_01 = RBFInterpolator(np.vstack((xpoints, xpoints)).T, ypoints, smoothing=0.1, kernel='cubic')

# Create an interpolant in casadi by evaluating the scipy one on an equal grid
x_interpolated = np.linspace(1, 6, 100)
y_interpolated = interp_func_2D_01(np.vstack((x_interpolated, x_interpolated)).T)
lut_2D_01 = cas.interpolant("lut", "bspline", [x_interpolated], y_interpolated)

# Plot that the spline fits not too badly the original data
# and that the scipy and casadi interpolants are exactly the same
x_interpolated = np.linspace(1, 6, 100)
axs[0].plot(xpoints, ypoints, 'ob')
axs[0].plot(x_interpolated, interp_func_2D_01(np.vstack((x_interpolated, x_interpolated)).T), '.y', label='scipy RBFInterpolator')
axs[0].plot(x_interpolated, x_interpolated * np.sin(5*x_interpolated) + x_interpolated**1/3, '-r', label='true function')
axs[0].plot(x_interpolated, lut_2D_01(x_interpolated), '--k', label='casadi interpolant')
axs[0].legend()


# --- Test 3D version --- #
zpoints = np.sin(xpoints) + ypoints + np.random.normal(0, 0.1, xpoints.shape)

# Create an interpolant in scipy
interp_func_3D_01 = RBFInterpolator(np.vstack((xpoints, ypoints)).T, zpoints, smoothing=0.1, kernel='cubic')

# Create an interpolant in casadi by evaluating the scipy one on an equal grid
x_interpolated = np.linspace(1, 6, 100)
y_interpolated = np.linspace(1, 6, 100)
xgrid, ygrid = np.meshgrid(x_interpolated, y_interpolated, indexing='ij')
z_interpolated = interp_func_3D_01(np.vstack((xgrid.ravel(), ygrid.ravel())).T)
lut_3D_01 = cas.interpolant("lut", "bspline", [x_interpolated, y_interpolated], z_interpolated)

# Plot that the spline fits not too badly the original data
# and that the scipy and casadi interpolants are exactly the same
surf1 = axs[1].plot_surface(xgrid, ygrid, z_interpolated.reshape(100, 100), color="r", alpha=0.8, label='scipy RBFInterpolator')
surf2 = axs[1].plot_surface(xgrid, ygrid, np.array(lut_3D_01(np.vstack((xgrid.ravel(order="F"), ygrid.ravel(order="F"))))).reshape(100, 100), color="b", alpha=0.8, label='casadi interpolant')
axs[1].scatter(xpoints, ypoints, zpoints, color='k', s=50, label='data points')
axs[1].legend()

plt.savefig("spline_interpolation_tests.png")
plt.show()
