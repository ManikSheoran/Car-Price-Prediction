from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained Random Forest model
model = joblib.load('trained_model.pkl')

# Define the correct order of features
feature_order = [
    'Kilometers_Driven', 'Owner_Type', 'Mileage', 'Engine', 'Power',
    'Seats', 'Age', 'Location_Ahmedabad', 'Location_Bangalore',
    'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
    'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
    'Location_Kolkata', 'Location_Mumbai', 'Location_Pune',
    'Fuel_Type_CNG', 'Fuel_Type_Diesel', 'Fuel_Type_LPG', 'Fuel_Type_Petrol',
    'Transmission_Manual'
]

@app.route('/')
def index():
    return render_template('./templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get features from the form
        kilometers_driven = float(request.form['kilometers_driven'])
        owner_type = request.form['owner_type']
        mileage = float(request.form['mileage'])
        engine = float(request.form['engine'])
        power = float(request.form['power'])
        seats = float(request.form['seats'])
        age = float(request.form['age'])  # Corrected 'age' feature name
        location = request.form['location']
        fuel_type = request.form['fuel_type']
        transmission = request.form['transmission']

        # Create DataFrame from user input
        input_data = pd.DataFrame({
            'Kilometers_Driven': [kilometers_driven],
            'Owner_Type': [owner_type],
            'Mileage': [mileage],
            'Engine': [engine],
            'Power': [power],
            'Seats': [seats],
            'Age': [age],  # Corrected 'Age' feature name
            'Location_Ahmedabad': [1 if location == 'Ahmedabad' else 0],
            'Location_Bangalore': [1 if location == 'Bangalore' else 0],
            'Location_Chennai': [1 if location == 'Chennai' else 0],
            'Location_Coimbatore': [1 if location == 'Coimbatore' else 0],
            'Location_Delhi': [1 if location == 'Delhi' else 0],
            'Location_Hyderabad': [1 if location == 'Hyderabad' else 0],
            'Location_Jaipur': [1 if location == 'Jaipur' else 0],
            'Location_Kochi': [1 if location == 'Kochi' else 0],
            'Location_Kolkata': [1 if location == 'Kolkata' else 0],
            'Location_Mumbai': [1 if location == 'Mumbai' else 0],
            'Location_Pune': [1 if location == 'Pune' else 0],
            'Fuel_Type_CNG': [1 if fuel_type == 'CNG' else 0],
            'Fuel_Type_Diesel': [1 if fuel_type == 'Diesel' else 0],
            'Fuel_Type_LPG': [1 if fuel_type == 'LPG' else 0],
            'Fuel_Type_Petrol': [1 if fuel_type == 'Petrol' else 0],
            'Transmission_Manual': [1 if transmission == 'Manual' else 0]
        })

        # Reorder the columns to match the model's feature order
        input_data = input_data[feature_order]

        # Make prediction
        predicted_price = model.predict(input_data)

        return render_template('./templates/result.html', prediction=predicted_price[0])


if __name__ == '__main__':
    app.run(debug=True)
