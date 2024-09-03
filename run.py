from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle
import sqlite3

app = Flask(__name__)

# Load the trained model and label encodings
model = pickle.load(open('best_random_forest_model.pkl', 'rb'))
label_encodings = pickle.load(open('label_encodings.pkl', 'rb'))

# Database setup
def init_db():
    conn = sqlite3.connect('contact_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS contacts
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  email TEXT NOT NULL,
                  message TEXT NOT NULL)''')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    return render_template('index.html', 
                           car_names=label_encodings['car_name'].keys(),
                           models=label_encodings['model'].keys(),
                           transmissions=label_encodings['transmission_type'].keys(),
                           sellers=label_encodings['seller_type'].keys(),
                           fuels=label_encodings['fuel_type'].keys())

@app.route('/analysis')
def car_analysis():
    return render_template('analysis.html')

@app.route('/predict', methods=['GET', 'POST'])
def price_predict():
    if request.method == 'POST':
        try:
            categorical_features = [
                request.form['car_name'],
                request.form['model_name'],
                request.form['transmission_type'],
                request.form['seller_type'],
                request.form['fuel_type']
            ]

            encoded_categorical_features = encode_features(categorical_features, label_encodings)

            # Get numerical inputs with limits
            max_power = int(request.form['max_power'])
            engine_size = int(request.form['engine_size'])
            mileage = float(request.form['mileage'])
            vehicle_age = int(request.form['vehicle_age'])
            km_driven = int(request.form['km_driven'])

            # Apply limits
            if max_power < 50 or max_power > 600:
                raise ValueError("Max Power should be between 50 and 600 HP")
            if engine_size < 500 or engine_size > 5000:
                raise ValueError("Engine size should be between 500 and 5000 CC")
            if mileage < 5 or mileage > 40:
                raise ValueError("Mileage should be between 5 and 40 KMPL")
            if vehicle_age < 0 or vehicle_age > 25:
                raise ValueError("Vehicle Age should be between 0 and 25 years")
            if km_driven < 0 or km_driven > 500000:
                raise ValueError("Kilometers driven should be between 0 and 500,000 KM")

            X_input = np.array([encoded_categorical_features + [max_power, engine_size, mileage, vehicle_age, km_driven]])

            y_pred = model.predict(X_input)

            return render_template('predict.html', 
                                   car_names=label_encodings['car_name'].keys(),
                                   models=label_encodings['model'].keys(),
                                   transmissions=label_encodings['transmission_type'].keys(),
                                   sellers=label_encodings['seller_type'].keys(),
                                   fuels=label_encodings['fuel_type'].keys(),
                                   prediction=f"Predicted Selling Price: ₹{y_pred[0]:,.2f}",
                                   error=None)
        except ValueError as e:
            return render_template('predict.html', 
                                   car_names=label_encodings['car_name'].keys(),
                                   models=label_encodings['model'].keys(),
                                   transmissions=label_encodings['transmission_type'].keys(),
                                   sellers=label_encodings['seller_type'].keys(),
                                   fuels=label_encodings['fuel_type'].keys(),
                                   prediction=None,
                                   error=str(e))
    return render_template('predict.html', 
                           car_names=label_encodings['car_name'].keys(),
                           models=label_encodings['model'].keys(),
                           transmissions=label_encodings['transmission_type'].keys(),
                           sellers=label_encodings['seller_type'].keys(),
                           fuels=label_encodings['fuel_type'].keys(),
                           prediction=None,
                           error=None)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        
        conn = sqlite3.connect('contact_data.db')
        c = conn.cursor()
        c.execute("INSERT INTO contacts (name, email, message) VALUES (?, ?, ?)",
                  (name, email, message))
        conn.commit()
        conn.close()
        
        return redirect(url_for('contact'))

    return render_template('contactus.html')

def encode_features(categorical_features, label_encodings):
    encoded_features = []
    for feature, encoding in zip(categorical_features, [label_encodings['car_name'], label_encodings['model'],
                                                       label_encodings['transmission_type'], label_encodings['seller_type'],
                                                       label_encodings['fuel_type']]):
        encoded_features.append(encoding.get(feature, -1))  # Use -1 for unknown categories
    return encoded_features

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
