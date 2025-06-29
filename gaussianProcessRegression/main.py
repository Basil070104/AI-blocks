import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def run():
  # Parameters for the bivariate normal
  mu = [0, 0] # u
  rho = 0.8 # 
  cov = [[1, rho], [rho, 1]]

  # Create grid
  x = np.linspace(-3, 3, 100)
  y = np.linspace(-3, 3, 100)
  X, Y = np.meshgrid(x, y)

  # Compute density
  inv_cov = np.linalg.inv(cov)
  det_cov = np.linalg.det(cov)
  norm_const = 1 / (2 * np.pi * np.sqrt(det_cov))

  Z = np.zeros_like(X)
  for i in range(X.shape[0]):
      for j in range(X.shape[1]):
          z = np.array([X[i, j], Y[i, j]]) - mu
          Z[i, j] = norm_const * np.exp(-0.5 * z.T.dot(inv_cov).dot(z))

  # 3D surface plot
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(projection='3d')
  ax1.plot_surface(X, Y, Z)
  ax1.set_title('3D Surface of Bivariate Normal PDF')
  ax1.set_xlabel('$x_1$')
  ax1.set_ylabel('$x_2$')
  ax1.set_zlabel('Density')

  # 2D contour plot
  fig2 = plt.figure()
  ax2 = fig2.add_subplot()
  ax2.contour(X, Y, Z, levels=6)
  ax2.set_title('Contour Lines of Bivariate Normal PDF')
  ax2.set_xlabel('$x_1$')
  ax2.set_ylabel('$x_2$')

  plt.show()


if __name__ == "__main__":
  run()