from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('iris_model.pkl')

# Route for home page (form input)
@app.route('/')
def index():
    return render_template('index.html')

# Route for form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    
    # Prepare the features for prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make the prediction using the loaded model
    prediction = model.predict(features)
    
    # Map the prediction to the actual iris species names
    iris_species = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    predicted_species = iris_species[prediction[0]]
    
    # Render the result template with the prediction
    return render_template('result.html', prediction=predicted_species)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

