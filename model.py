import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings

# Set display options
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# Ignore warnings
warnings.simplefilter(action='ignore')

# Load data
df = pd.read_csv('train-data-final.csv')

# Drop unnecessary columns
df.drop(['Unnamed: 0', 'Name', 'New_Price'], axis=1, inplace=True)

# Drop rows with missing values
df.dropna(subset=['Mileage', 'Engine', 'Power', 'Seats'], inplace=True)

# Convert 'Year' to 'Age'
df['Age'] = 2024 - df['Year']

# Encode categorical variables
Location = pd.get_dummies(df['Location'])
Fuel_t = pd.get_dummies(df['Fuel_Type'], prefix='Fuel_Type')
Transmission = pd.get_dummies(df['Transmission'], drop_first=True)

# Concatenate encoded features to the dataframe
df = pd.concat([df, Location, Fuel_t, Transmission], axis=1)

# Drop original categorical columns and 'Year'
df.drop(['Location', 'Fuel_Type', 'Transmission', 'Year'], axis=1, inplace=True)

# Define features and target variable
X = df[['Kilometers_Driven', 'Owner_Type', 'Mileage', 'Engine', 'Power',
        'Seats', 'Age', 'Fuel_Type_CNG', 'Fuel_Type_Diesel',
        'Fuel_Type_LPG', 'Fuel_Type_Petrol']]
y = df['Price']

# Initialize RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X, y)


# Function to get user input and predict car price
# Function to get user input and predict car price
def predict_car_price(model):
        print("Please enter the following details to get the predicted price:")

        kilometers_driven = float(input("Enter Kilometers Driven (default 0): ") or 0)
        owner_type = int(
                input("Enter Owner Type (0: First, 1: Second, 2: Third, 3: Fourth & Above) (default 0): ") or 0)
        mileage = float(input("Enter Mileage (kmpl) (default 0): ") or 0)
        engine = float(input("Enter Engine Displacement (CC) (default 0): ") or 0)
        power = float(input("Enter Power (bhp) (default 0): ") or 0)
        seats = int(input("Enter Number of Seats (default 0): ") or 0)
        year = int(input("Enter Manufacture Year (default 0): ") or 0)
        age = 2024 - year if year != 0 else 0

        print("Select Location:")
        print("1: Ahmedabad")
        print("2: Bangalore")
        print("3: Chennai")
        print("4: Coimbatore")
        print("5: Delhi")
        print("6: Hyderabad")
        print("7: Jaipur")
        print("8: Kochi")
        print("9: Kolkata")
        print("10: Mumbai")
        print("11: Pune")
        location_index = int(input("Enter Location Number (default 1): ") or 1) - 1

        fuel_type = input("Enter Fuel Type (Petrol, Diesel, CNG, LPG) (default Petrol): ") or 'Petrol'
        fuel_type_cng = 1 if fuel_type == 'CNG' else 0
        fuel_type_diesel = 1 if fuel_type == 'Diesel' else 0
        fuel_type_lpg = 1 if fuel_type == 'LPG' else 0
        fuel_type_petrol = 1 if fuel_type == 'Petrol' else 0

        transmission = input("Enter Transmission Type (Manual, Automatic) (default Manual): ") or 'Manual'
        transmission = 1 if transmission == 'Automatic' else 0

        # Create input data
        input_data = pd.DataFrame({
                'Kilometers_Driven': [kilometers_driven],
                'Owner_Type': [owner_type],
                'Mileage': [mileage],
                'Engine': [engine],
                'Power': [power],
                'Seats': [seats],
                'Age': [age],
                'Location': [location_index],
                'Fuel_Type_CNG': [fuel_type_cng],
                'Fuel_Type_Diesel': [fuel_type_diesel],
                'Fuel_Type_LPG': [fuel_type_lpg],
                'Fuel_Type_Petrol': [fuel_type_petrol],
                'Transmission': [transmission]
        })

        # Predict car price
        predicted_price = model.predict(input_data)[0]

        print(f"\nPredicted Car Price: {predicted_price} Lakh(s)")


# Predict car price using user input
predict_car_price(rf)

