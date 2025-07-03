import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import cholesky, solve_triangular

def rbf_kernel(X1, X2, length_scale=1.0, amplitude=1.0):
    # Vectorized RBF kernel
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return amplitude * np.exp(-0.5 * sqdist / length_scale**2)

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
  
def gp():
    # 1. Simulate some observed data
    X_train = np.array([[-4], [-2], [0], [2], [4]])
    y_train = np.sin(X_train) + 0.1 * np.random.randn(*X_train.shape)

    # 2. Test points (for prediction)
    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

    # 3. Compute kernel matrices
    K = rbf_kernel(X_train, X_train) + 1e-6 * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_test)
    K_ss = rbf_kernel(X_test, X_test) + 1e-6 * np.eye(len(X_test))

    # 4. Compute GP posterior mean and covariance
    L = cholesky(K, lower=True)
    alpha = solve_triangular(L.T, solve_triangular(L, y_train, lower=True), lower=False)
    mu_post = K_s.T @ alpha
    v = solve_triangular(L, K_s, lower=True)
    cov_post = K_ss - v.T @ v
    std_post = np.sqrt(np.diag(cov_post))

    # 5. Plot
    plt.figure(figsize=(10, 6))
    # Plot posterior mean
    plt.plot(X_test, mu_post, 'b', lw=2, label='Posterior Mean')
    # Plot uncertainty
    plt.fill_between(X_test.ravel(), mu_post.ravel() - 2*std_post, mu_post.ravel() + 2*std_post, color='blue', alpha=0.2, label='Uncertainty (2 std)')
    # Plot observed data
    plt.scatter(X_train, y_train, color='red', marker='o', label='Observed Data')
    # Plot samples from the posterior
    samples = np.random.multivariate_normal(mu_post.ravel(), cov_post, 3)
    for i, s in enumerate(samples):
        plt.plot(X_test, s, lw=1, ls='--', label=f'Posterior Sample {i+1}' if i==0 else None)
    plt.xlabel("Input (X)")
    plt.ylabel("Output (f(X))")
    plt.title("Gaussian Process Regression (Posterior)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
  gp()