from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# === Step 1: Train a simple regression model (once) ===
def train_model():
    # Dummy dataset: hours, previous_score, attendance, actual_score
    X = np.array([
        [2, 50, 70],
        [5, 60, 80],
        [8, 65, 90],
        [3, 40, 60],
        [7, 75, 95],
        [6, 70, 85]
    ])
    y = np.array([55, 68, 85, 50, 90, 80])

    model = LinearRegression()
    model.fit(X, y)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

try:
    open('model.pkl', 'rb')
except FileNotFoundError:
    train_model()

# === Step 2: Load the model and predict ===
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    prev_score = float(request.form['previous_score'])
    attendance = float(request.form['attendance'])
    features = np.array([[hours, prev_score, attendance]])

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(features)[0]
    prediction = round(prediction, 2)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
