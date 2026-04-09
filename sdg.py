import numpy as np
from dataGen import generate_data

#Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_gradient(w, x_tilde, y):
    #Compute gradient of logistic loss with respect to w
    z = -y * np.dot(w, x_tilde)
    grad = -y * sigmoid(z) * x_tilde
    return grad

def project_unit_ball(w):
    #Project w onto the unit ball after each step
    norm = np.linalg.norm(w)
    return w / norm if norm > 1 else w

def logistic_loss(w, X, Y):
    #Compute average logistic loss of w over a dataset
    n = len(Y)
    total = 0.0
    for i in range(n):
        x_tilde = np.append(X[i], 1)
        z = -Y[i] * np.dot(w, x_tilde)
        total += np.log(1 + np.exp(z))
    return total / n


def classification_error(w, X, Y):
    #Calculate fraction of misclassified examples
    n = len(Y)
    mistakes = 0
    for i in range(n):
        x_tilde = np.append(X[i], 1)
        prediction = np.sign(np.dot(w, x_tilde))
        if prediction != Y[i]:
            mistakes += 1
    return mistakes / n

def sgd_logistic(X_train, Y_train, rho=np.sqrt(2), M=1.0):
    #Run SGD for logistic regression over the training set.
    #One pass through the data = one run of SGD with n iterations.

    n, d = X_train.shape
    w = np.zeros(d + 1)   #initialize at origin (inside unit ball)
    T = n                  #total number of iterations

    for t in range(1, T + 1):
        #Pick training example at index t-1 (already in order, which is fine for SGD)
        x_tilde = np.append(X_train[t-1], 1)
        y = Y_train[t-1]

        #Compute step size
        eta_t = rho * M / np.sqrt(T)   #constant step size variant (DOUBLE CHECK THIS WITH NOTES) η_t = 1/√t or η_t = ρM / (‖g_t‖ · √T)

        #Gradient step
        grad = logistic_gradient(w, x_tilde, y)
        w = w - eta_t * grad

        #Project back onto unit ball
        w = project_unit_ball(w)

    return w

#Test the functions
X_train, Y_train = generate_data(n=1000, sigma=0.2)
X_test, Y_test   = generate_data(n=400,  sigma=0.2)

for n in [50, 100, 500, 1000]:
    w = sgd_logistic(X_train[:n], Y_train[:n])
    loss  = logistic_loss(w, X_test, Y_test)
    error = classification_error(w, X_test, Y_test)
    print(f"n={n:4d} | Loss: {loss:.4f} | Error: {error:.4f}")