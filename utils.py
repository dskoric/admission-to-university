import matplotlib.pyplot as plt
import numpy as np

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.
    
    Parameters
    ----------
    X : array_like
        An Mx2 matrix representing the dataset. 
    y : array_like
        Label values for the dataset. A vector of size (M, ).
    
    Returns
    -------
    None
    """
    # Create a new figure
    plt.figure(figsize=(10, 7))
    
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    
    # Plot examples
    plt.plot(X[pos, 0], X[pos, 1], 'k*', markersize=12, label=pos_label)
    plt.plot(X[neg, 0], X[neg, 1], 'yo', markersize=10, label=neg_label)
    
    # Set the y-axis label
    plt.ylabel('Exam 2 score')
    # Set the x-axis label
    plt.xlabel('Exam 1 score')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()