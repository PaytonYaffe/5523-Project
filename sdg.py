
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
    w = np.zeros(d + 1)       #initialize at origin (inside unit ball)
    T = n                      #total number of iterations
    w_sum = np.zeros(d + 1)   #accumulate w values for averaging
 
    for t in range(1, T + 1):
        x_tilde = np.append(X_train[t-1], 1)  #augment with bias term
        y = Y_train[t-1]
 
        eta_t = M / (rho * np.sqrt(T))  #step size: M / (rho * sqrt(T))
 
        grad = logistic_gradient(w, x_tilde, y)
        w = w - eta_t * grad
        w = project_unit_ball(w)   #project back onto C
        w_sum += w                 #accumulate for average
 
    return w_sum / T  #return average predictor per SGD convergence theorem