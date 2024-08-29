from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model and label encodings
model = pickle.load(open('best_random_forest_model.pkl', 'rb'))
label_encodings = pickle.load(open('label_encodings.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', 
                           car_names=label_encodings['car_name'].keys(),
                           models=label_encodings['model'].keys(),
                           transmissions=label_encodings['transmission_type'].keys(),
                           sellers=label_encodings['seller_type'].keys(),
                           fuels=label_encodings['fuel_type'].keys())

@app.route('/car-analysis')
def car_analysis():
    return render_template('analysis.html')

@app.route('/price-predict')
def price_predict():
    return render_template('predict.html', 
                           car_names=label_encodings['car_name'].keys(),
                           models=label_encodings['model'].keys(),
                           transmissions=label_encodings['transmission_type'].keys(),
                           sellers=label_encodings['seller_type'].keys(),
                           fuels=label_encodings['fuel_type'].keys())

@app.route('/predict', methods=['POST'])
def predict():
    categorical_features = [
        request.form['car_name'],
        request.form['model_name'],
        request.form['transmission_type'],
        request.form['seller_type'],
        request.form['fuel_type']
    ]

    encoded_categorical_features = encode_features(categorical_features, label_encodings)

    max_power = int(request.form['max_power'])
    engine = int(request.form['engine'])
    mileage = float(request.form['mileage'])
    vehicle_age = int(request.form['vehicle_age'])
    km_driven = int(request.form['km_driven'])

    X_input = np.array([encoded_categorical_features + [max_power, engine, mileage, vehicle_age, km_driven]])

    y_pred = model.predict(X_input)

    return f"Predicted Selling Price: ₹{y_pred[0]:,.2f}"

@app.route('/contact')
def contact():
    return "Contact Us Page"

def encode_features(categorical_features, label_encodings):
    encoded_features = []
    for feature, encoding in zip(categorical_features, [label_encodings['car_name'], label_encodings['model'],
                                                       label_encodings['transmission_type'], label_encodings['seller_type'],
                                                       label_encodings['fuel_type']]):
        encoded_features.append(encoding.get(feature, -1))  # Use -1 for unknown categories
    return encoded_features

if __name__ == "__main__":
    app.run(debug=True)
