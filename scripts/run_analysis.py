import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_loader import load_and_preprocess_data
from models import LogisticRegression
from visualization import plot_decision_boundary, plot_data, plot_cost_history

def main():
    # Load data
    data_path = os.path.join("data", "raw", "ex2data1.txt")
    X, y = load_and_preprocess_data(data_path)
    
    # Plot the original data
    plot_data(X, y)
    
    # Create and train model
    model = LogisticRegression(learning_rate=0.01, num_iterations=1000)
    cost_history = model.fit(X, y)
    
    # Plot cost history
    plot_cost_history(cost_history)
    
    # Evaluate model
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Visualize results with decision boundary
    plot_decision_boundary(X, y, model)

if __name__ == "__main__":
    main()