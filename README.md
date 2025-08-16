# University Admission Prediction using Logistic Regression

This project predicts university admission chances based on student academic profiles using logistic regression analysis.

## Project Structure

├── README.md # Project documentation
├── .gitignore # Files to ignore in Git
├── requirements.txt # Python dependencies
├── main.py # Main entry point
├── notebooks/
│ └── logistic-regression.ipynb # Original notebook analysis
├── src/ # Source code
│ ├── data_processing.py # Data loading and preprocessing
│ ├── logistic_regression.py # Logistic regression implementation
│ ├── model_evaluation.py # Model evaluation metrics
│ └── visualization.py # Plotting functions
├── data/ # Data files
│ ├── raw/ # Raw dataset
│ └── processed/ # Processed dataset
├── tests/ # Unit tests
├── models/ # Trained model files
└── results/ # Evaluation results and plots


## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/dskoric/admission-to-university.git
cd admission-to-university

2. Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
pip install -r requirements.txt

Usage
1. Run the entire pipeline:
python main.py

2. Run the Jupyter notebook for exploratory data analysis:
jupyter notebook notebooks/logistic-regression.ipynb


jupyter notebook notebooks/logistic-regression.ipynb

#Dataset Description
The dataset contains information about students including:

GRE Scores
TOEFL Scores
University Rating
Statement of Purpose (SOP)
Letter of Recommendation (LOR)
Undergraduate GPA (CGPA)
Research experience
Chance of Admit (target variable)
Results
The logistic regression model achieves an accuracy of approximately 85% in predicting university admission chances.