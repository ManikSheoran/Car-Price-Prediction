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
    default_values = {
        'Kilometers_Driven': 20000,
        'Year': 2020,
        'Owner_Type': 1,
        'Mileage': 18,
        'Engine': 1100,
        'Power': 70,
        'Seats': 5,
        'Fuel_Type': 'Petrol',
        'Transmission': 'Manual',
        'Location': 'Kolkata'
    }

    input_df = pd.DataFrame({
        'Kilometers_Driven': [float(input_data.get('Kilometers_Driven', default_values['Kilometers_Driven'])) if input_data.get('Kilometers_Driven', '') != '' else float(default_values['Kilometers_Driven'])],
        'Year': [int(input_data.get('Year', default_values['Year'])) if input_data.get('Year', '') != '' else int(default_values['Year'])],
        'Owner_Type': [int(input_data.get('Owner_Type', default_values['Owner_Type']))],
        'Mileage': [float(input_data.get('Mileage', f"{default_values['Mileage']} kmpl").split()[0]) if input_data.get('Mileage', '') != '' else float(default_values['Mileage'])],
        'Engine': [float(input_data.get('Engine', f"{default_values['Engine']} CC").split()[0]) if input_data.get('Engine', '') != '' else float(default_values['Engine'])],
        'Power': [float(input_data.get('Power', f"{default_values['Power']} bhp").split()[0]) if input_data.get('Power', '') != '' else float(default_values['Power'])],
        'Seats': [float(input_data.get('Seats', default_values['Seats'])) if input_data.get('Seats', '') != '' else float(default_values['Seats'])],
        'Fuel_Type_CNG': [1 if input_data.get('Fuel_Type', default_values['Fuel_Type']) == 'CNG' else 0],
        'Fuel_Type_Diesel': [1 if input_data.get('Fuel_Type', default_values['Fuel_Type']) == 'Diesel' else 0],
        'Fuel_Type_LPG': [1 if input_data.get('Fuel_Type', default_values['Fuel_Type']) == 'LPG' else 0],
        'Fuel_Type_Petrol': [1 if input_data.get('Fuel_Type', default_values['Fuel_Type']) == 'Petrol' else 0],
        'Transmission_Manual': [1 if input_data.get('Transmission', default_values['Transmission']) == 'Manual' else 0],
        'Location_Ahmedabad': [1 if input_data.get('Location', default_values['Location']) == 'Ahmedabad' else 0],
        'Location_Bangalore': [1 if input_data.get('Location', default_values['Location']) == 'Bangalore' else 0],
        'Location_Chennai': [1 if input_data.get('Location', default_values['Location']) == 'Chennai' else 0],
        'Location_Coimbatore': [1 if input_data.get('Location', default_values['Location']) == 'Coimbatore' else 0],
        'Location_Delhi': [1 if input_data.get('Location', default_values['Location']) == 'Delhi' else 0],
        'Location_Hyderabad': [1 if input_data.get('Location', default_values['Location']) == 'Hyderabad' else 0],
        'Location_Jaipur': [1 if input_data.get('Location', default_values['Location']) == 'Jaipur' else 0],
        'Location_Kochi': [1 if input_data.get('Location', default_values['Location']) == 'Kochi' else 0],
        'Location_Kolkata': [1 if input_data.get('Location', default_values['Location']) == 'Kolkata' else 0],
        'Location_Mumbai': [1 if input_data.get('Location', default_values['Location']) == 'Mumbai' else 0],
        'Location_Pune': [1 if input_data.get('Location', default_values['Location']) == 'Pune' else 0]
    })


    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)

    return render_template('result.html', prediction=abs(round(prediction[0], 2)))


if __name__ == '__main__':
    app.run(debug=True)
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=5000)
