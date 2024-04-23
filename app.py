from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('catboost_model.pkl')
scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    input_data = request.form.to_dict()

    # Prepare the input data as a DataFrame
    input_df = pd.DataFrame({
        'Kilometers_Driven': [float(input_data.get('Kilometers_Driven', 20000))],
        'Year': [int(input_data.get('Year', 2020))],
        'Owner_Type': [int(input_data.get('Owner_Type', 1))],
        'Mileage': [float(input_data.get('Mileage', '18 kmpl').split()[0]) if input_data.get('Mileage') else float(18)],
        'Engine': [float(input_data.get('Engine', '1100 CC').split()[0]) if input_data.get('Engine') else float(1100)],
        'Power': [float(input_data.get('Power', '70 bhp').split()[0]) if input_data.get('Power') else float(70)],
        'Seats': [float(input_data.get('Seats', 5))],
        'Fuel_Type_CNG': [1 if input_data.get('Fuel_Type', '') == 'CNG' else 0],
        'Fuel_Type_Diesel': [1 if input_data.get('Fuel_Type', '') == 'Diesel' else 0],
        'Fuel_Type_LPG': [1 if input_data.get('Fuel_Type', '') == 'LPG' else 0],
        'Fuel_Type_Petrol': [1 if input_data.get('Fuel_Type', '') == 'Petrol' else 0],
        'Transmission_Manual': [1 if input_data.get('Transmission', '') == 'Manual' else 0],
        'Location_Ahmedabad': [1 if input_data.get('Location', '') == 'Ahmedabad' else 0],
        'Location_Bangalore': [1 if input_data.get('Location', '') == 'Bangalore' else 0],
        'Location_Chennai': [1 if input_data.get('Location', '') == 'Chennai' else 0],
        'Location_Coimbatore': [1 if input_data.get('Location', '') == 'Coimbatore' else 0],
        'Location_Delhi': [1 if input_data.get('Location', '') == 'Delhi' else 0],
        'Location_Hyderabad': [1 if input_data.get('Location', '') == 'Hyderabad' else 0],
        'Location_Jaipur': [1 if input_data.get('Location', '') == 'Jaipur' else 0],
        'Location_Kochi': [1 if input_data.get('Location', '') == 'Kochi' else 0],
        'Location_Kolkata': [1 if input_data.get('Location', '') == 'Kolkata' else 0],
        'Location_Mumbai': [1 if input_data.get('Location', '') == 'Mumbai' else 0],
        'Location_Pune': [1 if input_data.get('Location', '') == 'Pune' else 0]
    })


    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    return render_template('result.html', prediction=abs(round(prediction[0], 2)))


if __name__ == '__main__':
    app.run(debug=True)
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
