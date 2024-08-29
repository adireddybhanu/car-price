import streamlit as st
import numpy as np
import pickle

# Load the trained model
def load_model():
    with open('best_random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the label encodings
def load_label_encodings():
    with open('label_encodings.pkl', 'rb') as file:
        label_encodings = pickle.load(file)
    return label_encodings

# Encode categorical features
def encode_features(categorical_features, label_encodings):
    encoded_features = []
    for feature, encoding in zip(categorical_features, [label_encodings['car_name'], label_encodings['model'],
                                                       label_encodings['transmission_type'], label_encodings['seller_type'],
                                                       label_encodings['fuel_type']]):
        encoded_features.append(encoding.get(feature, -1))  # Use -1 for unknown categories
    return encoded_features

# Streamlit app
def main():
    st.title("Selling Price Prediction")

    # Load model and label encodings
    model = load_model()
    label_encodings = load_label_encodings()

    # Input fields for the categorical features
    car_name = st.selectbox("Select the car name:", options=label_encodings['car_name'].keys())
    model_name = st.selectbox("Select the model:", options=label_encodings['model'].keys())
    transmission_type = st.selectbox("Select the transmission type:", options=label_encodings['transmission_type'].keys())
    seller_type = st.selectbox("Select the seller type:", options=label_encodings['seller_type'].keys())
    fuel_type = st.selectbox("Select the fuel type:", options=label_encodings['fuel_type'].keys())

    # Input fields for the numerical features with limits
    max_power = st.number_input("Enter the max power (e.g., 150):", min_value=30, max_value=600, value=150)
    engine = st.number_input("Enter the engine size (e.g., 1500):", min_value=500, max_value=5000, value=1500)
    mileage = st.number_input("Enter the mileage (e.g., 18):", min_value=4.0, max_value=33.0, value=18.0)
    vehicle_age = st.number_input("Enter the vehicle age (e.g., 3):", min_value=0, max_value=33, value=3)
    km_driven = st.number_input("Enter the kilometers driven (e.g., 30000):", min_value=100, max_value=380000, value=30000)

    # Encode the categorical features
    categorical_features = [car_name, model_name, transmission_type, seller_type, fuel_type]
    encoded_categorical_features = encode_features(categorical_features, label_encodings)

    # Combine encoded categorical features with numerical features
    X_input = np.array([encoded_categorical_features + [max_power, engine, mileage, vehicle_age, km_driven]])

    if st.button("Predict Selling Price"):
        # Predict the selling price
        y_pred = model.predict(X_input)

        st.success(f"Predicted Selling Price: ₹{y_pred[0]:,.2f}")

if __name__ == "__main__":
    main()
