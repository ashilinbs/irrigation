from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model, scaler, and label encoder
model = joblib.load('final_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Extract input features
    crop_type = data['CropType']
    crop_days = data['CropDays']
    soil_moisture = data['SoilMoisture']
    temperature = data['Temperature']
    humidity = data['Humidity']

    # Encode and scale input data
    crop_type_encoded = label_encoder.transform([crop_type])[0]  # Convert crop type to numerical form
    input_data = np.array([[crop_days, soil_moisture, temperature, humidity]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)
    irrigation_status = 'Irrigation Needed' if prediction[0] == 1 else 'No Irrigation Needed'

    # Return the result
    return jsonify({"IrrigationStatus": irrigation_status})

if __name__ == '__main__':
    app.run(debug=True)
