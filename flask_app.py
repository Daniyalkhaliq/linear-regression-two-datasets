from flask import Flask, render_template, request
from train_model import DATASETS, load_and_train

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    details = None
    error = None
    selected_dataset = "Salary Prediction"
    input_value = ""
    if request.method == "POST":
        selected_dataset = request.form.get("dataset", "Salary Prediction")
        input_value = request.form.get("input_value", "")
        try:
            model, metrics = load_and_train(selected_dataset)
            number = float(input_value)
            prediction_value = float(model.predict([[number]])[0])
            prediction = f"{metrics['target_label']}: {prediction_value:,.2f}"
            details = (
                f"Rows used: {metrics['rows_used']} | "
                f"MAE: {metrics['MAE']:.2f} | RMSE: {metrics['RMSE']:.2f} | "
                f"R²: {metrics['R²']:.3f}\n"
                f"Model equation: y = {metrics['slope']:.3f}x + {metrics['intercept']:.3f}"
            )
        except Exception as exc:
            error = str(exc)
    return render_template(
        "index.html",
        datasets=list(DATASETS.keys()),
        selected_dataset=selected_dataset,
        input_value=input_value,
        prediction=prediction,
        details=details,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
