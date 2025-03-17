from flask import Flask, request, jsonify
import joblib
import numpy as np

model = joblib.load('model/eunoia_model_ghq12.pkl')
scaler = joblib.load('model/eunoia_scaler_ghq12.pkl')
label_encoder = joblib.load('model/eunoia_label_encoder_ghq12.pkl')

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:

        print("Received request:", request.get_json())
        # Get the request JSON data
        data = request.get_json()

        # Extract values
        ghq12_responses = data['ghq12_responses']
        age = data['age']
        gender = data['gender']

        # Reverse scoring for positive questions
        positive_questions = ["Q1", "Q3", "Q4", "Q7", "Q8", "Q12"]
        for i, value in enumerate(ghq12_responses):
            if f"Q{i + 1}" in positive_questions:
                ghq12_responses[i] = 3 - value


        # Manual Gender Conversion from Male -> 0, Female -> 1
        gender_encoded = 0 if gender.lower() == "male" else 1 if gender.lower() == "female" else None
        if gender_encoded is None:
            raise ValueError("Invalid gender value. Please enter 'Male' or 'Female'.")

        # Encode gender
        #gender_encoded = label_encoder.transform([gender])[0]

        # Prepare input for the model
        user_input = ghq12_responses + [age, gender_encoded]
        user_input = np.array(user_input).reshape(1, -1)

        # Scale the input
        scaled_input = scaler.transform(user_input)

        # Predict using the model
        prediction = model.predict(scaled_input)[0]

        # Return the prediction as JSON
        return jsonify({'recommended_professional': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
