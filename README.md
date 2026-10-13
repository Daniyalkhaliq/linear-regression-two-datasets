# Linear Regression on Two Datasets

A beginner-friendly Python project that trains **two separate Linear Regression models** and provides two local web interfaces:
- **Gradio** interface (`gradio_app.py`)
- **Flask** interface (`flask_app.py`)

## Datasets

1. **Salary Prediction** — input: years of experience; target: salary.
2. **Student Exam Score** — input: hours studied; target: exam score.

Download CSV datasets from Kaggle and put them in the `datasets` folder using these exact filenames:
- `salary_data.csv`
- `student_performance.csv`

The code expects columns similar to:
- Salary CSV: `YearsExperience` (or `Years of Experience`) and `Salary`
- Student CSV: `Hours_Studied` (or `Hours Studied`) and `Exam_Score` (or `Exam Score`)

If the column names differ, either rename the columns in the CSV or update the candidate names in `train_model.py`.

## Setup (Windows)

1. Install Python 3.10 or newer from https://www.python.org/downloads/ and select **Add Python to PATH** during installation.
2. Extract/download this project folder.
3. Open the folder in File Explorer.
4. Click the address bar, type `cmd`, and press Enter.
5. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
6. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
7. Put the two CSV files in `datasets/`.

## Run model evaluation

```bash
python train_model.py
```

This prints Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) for each dataset.

## Run the Gradio interface

```bash
python gradio_app.py
```

Open the local address printed in the terminal (usually http://127.0.0.1:7860).

## Run the Flask interface

Open a **second terminal** in the project folder, activate the virtual environment again, then run:

```bash
python flask_app.py
```

Open http://127.0.0.1:5000 in your browser. Stop a running app with Ctrl+C in its terminal.

## How it works

For each dataset, the program:
1. Loads the CSV file with pandas.
2. Selects one numeric input feature and one numeric target.
3. Removes rows with missing or non-numeric values in those columns.
4. Splits data into 80% training and 20% testing data.
5. Fits `sklearn.linear_model.LinearRegression`.
6. Reports MAE, RMSE and R² on the test set.
7. Predicts the target for the value entered in the web interface.

## Important

- This is a learning demonstration using **one input feature per dataset** so the regression and interface stay simple.
- Results depend on the downloaded datasets; do not claim any score before running the code.
- Check each Kaggle dataset's license and attribution requirements before redistributing the data.
- The CSV files are not included in this starter project; add them locally after downloading from Kaggle.
