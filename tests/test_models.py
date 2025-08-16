import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models import LogisticRegression

def test_logistic_regression():
    # Create simple test data
    X = np.array([[1, 2], [2, 3], [3, 1], [4, 5], [5, 4]])
    y = np.array([0, 0, 0, 1, 1])
    
    # Train model
    model = LogisticRegression(learning_rate=0.1, num_iterations=100)
    model.fit(X, y)
    
    # Test predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    
    # Test accuracy
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.5  # Should perform better than random