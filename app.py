from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the trained SVM models
def load_models(departments):
    trained_models = {}
    for department in departments:
        filename = f"{department}_svm_model.joblib"
        model = load(filename)
        trained_models[department] = model
    return trained_models

# List of departments
departments = ['Computer Engineering', 'Civil Engineering', 'Mechanical Engineering', 'Electrical Engineering', 'Chemical Engineering']
# Load the trained models
trained_models = load_models(departments)

# Index route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        department = request.form['department']
        year = int(request.form['year'])
        # Make predictions for the selected department and year
        predictions = make_predictions_future_year(trained_models[department], year)
        return render_template('result.html', department=department, year=year, predictions=predictions)
    return render_template('index.html', departments=departments)

# Function to make predictions for the future year using loaded models
def make_predictions_future_year(model, future_year):
    prediction = model.predict([[future_year]])
    columns = ['Total_Admission_Count', 'SC', 'GEN-EWS', 'ST', 'PwD', 'OBC', 'GEN', 'Navsari', 'Bardoli', 'Maroli', 'Surat', 'Chikhli', 'Bilimora', 'Valsad', 'Dharampur', 'Vapi', 'Other_Area']
    return {columns[i]: prediction[0][i] for i in range(len(columns))}
if __name__ == '__main__':
    app.run(debug=True)
