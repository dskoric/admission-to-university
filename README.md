# University Admission Prediction using Logistic Regression

This project predicts university admission chances based on student academic profiles using logistic regression analysis.

## Project Structure

admission-to-university/
├── data/ # Dataset
├── notebooks/ # Jupyter notebooks
├── src/ # Python modules
├── scripts/ # Analysis scripts
├── requirements.txt # Dependencies
└── .gitignore # Files to ignore

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: `python scripts/run_analysis.py`

## Dataset
The dataset contains applicant information including:
- GRE scores
- TOEFL scores
- University rating
- Statement of Purpose strength
- Letter of Recommendation strength
- Undergraduate GPA
- Research experience
- Chance of admission (target variable)

## Results
The logistic regression model achieves 92% accuracy in predicting admission chances.
