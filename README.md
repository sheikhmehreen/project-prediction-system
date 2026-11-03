# Project Overrun Predictor

An enterprise-grade predictive system built to forecast software project timeline and effort overruns using robust machine learning algorithms. This system implements production-level validation frameworks to eliminate data leakage and support accurate cost optimization assessments for software and IT operations.

## Tech Stack & Architecture
* **Language & Runtime:** Python 3.11+
* **Core Frameworks:** Scikit-Learn, Pandas, NumPy
* **Notebook Environments:** Jupyter (`stage1.ipynb`, `stage2.ipynb`)
* **Dependency Orchestration:** Pip (`requirements.txt`)

## Cross-Validation Strategy
To guarantee stable generalization on historically small datasets (such as NASA93), this system avoids native random splits and implements a strict **Nested Cross-Validation** strategy:
* **Outer Loop (Generalization Split):** Evaluates performance boundaries against completely unseen data folds.
* **Inner Loop (Hyperparameter Isolation):** Executes hyperparameter optimization via `GridSearchCV` completely isolated within the training block to prevent performance optimism or data leakage.

## Model Performance & Production Baseline
Evaluation results across classification iterations validated against a structural `DummyClassifier` baseline:

| Model Architecture | Mean Accuracy | Mean F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Gradient Boosting Classifier** | **0.6340** | **0.6216** | **0.7003** | Selected Production Model |
| Logistic Regression | 0.5820 | 0.5510 | 0.6120 | Baseline |
| Random Forest | 0.6110 | 0.5980 | 0.6640 | Baseline |
| Dummy Baseline | 0.5000 | 0.4800 | 0.5000 | Reference |

## Environment Setup
Reconstruct the local isolated environment boundaries without syncing binary cache allocations:
```bash
# Initialize local virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\Activate.ps1

# Install exact validated production dependencies
pip install -r requirements.txt
```
