import numpy as np

def generate_data(n, sigma, d=4, seed=None):
    """
    Generate n examples from the data distribution D.

    Each example (x, y) is generated as:
      -y is drawn uniformly from {-1, +1}
      -u is drawn from N(mu_y, sigma^2 * I) where
          mu_{+1} = (1/4, 1/4, 1/4, 1/4)
          mu_{-1} = (-1/4, -1/4, -1/4, -1/4)
      -x = projection of u onto the unit ball in R^d
          i.e. x = u if ||u|| <= 1, else x = u / ||u||

    Parameters:
        n: number of examples to generate
        sigma: standard deviation of the Gaussian
        d: feature dimensionality (default 4)
        seed: optional random seed for reproducibility

    Returns:
        X : numpy array of shape (n, d) which are the feature vectors
        Y : numpy array of shape (n,) which are the labels in {-1, +1}
    """

    if seed is not None:
        np.random.seed(seed)

    mu_pos = np.full(d,  0.25)   #mean for y = +1
    mu_neg = np.full(d, -0.25)   #mean for y = -1

    X = np.zeros((n, d))
    Y = np.zeros(n)

    for i in range(n):
        # Step 1: flip a coin for the label
        y = 1 if np.random.rand() > 0.5 else -1

        # Step 2: sample from the appropriate Gaussian
        mu = mu_pos if y == 1 else mu_neg
        u = np.random.normal(loc=mu, scale=sigma, size=d)

        # Step 3: project onto the unit ball
        norm = np.linalg.norm(u)
        x = u / norm if norm > 1 else u

        X[i] = x
        Y[i] = y

    return X, Y

#Testing the function here
X, Y = generate_data(n=500, sigma=0.2)

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Max norm of X:", np.max(np.linalg.norm(X, axis=1)))
print("Unique labels:", np.unique(Y))
print("Label balance:", np.mean(Y == 1))
print("Mean of X[Y==1]:", X[Y==1].mean(axis=0))
print("Mean of X[Y==-1]:", X[Y==-1].mean(axis=0)) 