import gradio as gr
import joblib

# Load the trained models
salary_model = joblib.load("models/salary_model.pkl")
student_model = joblib.load("models/student_model.pkl")


# Salary prediction function
def predict_salary(years_experience):
    prediction = salary_model.predict([[years_experience]])[0]
    return f"Predicted Salary: {prediction:,.2f}"


# Student exam score prediction function
def predict_exam_score(hours_studied):
    prediction = student_model.predict([[hours_studied]])[0]
    return f"Predicted Exam Score: {prediction:.2f}"


# Salary prediction interface
salary_interface = gr.Interface(
    fn=predict_salary,
    inputs=gr.Number(label="Years of Experience"),
    outputs=gr.Textbox(label="Predicted Salary"),
    title="Salary Prediction",
    description="Predict salary based on years of work experience."
)


# Student exam score interface
student_interface = gr.Interface(
    fn=predict_exam_score,
    inputs=gr.Number(label="Hours Studied"),
    outputs=gr.Textbox(label="Predicted Exam Score"),
    title="Student Exam Score Prediction",
    description="Predict exam score based on study hours."
)


# Combine both interfaces into one application
app = gr.TabbedInterface(
    [salary_interface, student_interface],
    ["Salary Prediction", "Student Exam Score"]
)


# Launch the application
if __name__ == "__main__":
    print("Models loaded. Starting Gradio...")
    app.launch(show_error=True, quiet=False)
