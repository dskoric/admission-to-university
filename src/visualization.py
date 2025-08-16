import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y):
    """Plot the original data points.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
    """
    # Find indices of positive and negative examples
    pos = y == 1
    neg = y == 0
    
    # Plot examples
    plt.figure(figsize=(10, 7))
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, markersize=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.grid(True)
    plt.title('University Admission Data')
    plt.show()

def plot_decision_boundary(X, y, model):
    """Plot the decision boundary for logistic regression.
    
    Args:
        X (numpy.ndarray): Feature matrix
        y (numpy.ndarray): Target vector
        model: Trained logistic regression model
    """
    # Plot the original data
    plot_data(X, y)
    
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))
    
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)
    
    # Plot the contour
    plt.figure(figsize=(10, 7))
    plt.contour(xx, yy, Z, [0.5], linewidths=2, colors='g')
    
    # Plot the original data points
    pos = y == 1
    neg = y == 0
    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, markersize=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=8)
    
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
    plt.grid(True)
    plt.title('University Admission with Decision Boundary')
    plt.show()

def plot_cost_history(cost_history):
    """Plot the cost function over iterations.
    
    Args:
        cost_history (list): History of cost values during training
    """
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(cost_history)), cost_history, 'r-')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost Function During Training')
    plt.grid(True)
    plt.show()