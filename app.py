from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model_path = 'menstrual_onset_model.pkl'  # Adjust to match your model path
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define feature names and categorical options
features = [
    "Age", "Fast_Food_Intake", "Physical_Activity", "Sugar_Intake_Per_Day",
    "Fiber_Intake_Per_Day", "Protein_Intake_Per_Day", "Processed_Food_Score",
    "Dietary_Pattern", "Socioeconomic_Status", "Environmental_Factors",
    "Region", "Industrialization_Level", "Nutrition_Access"
]

categorical_mappings = {
    "Dietary_Pattern": {"Vegan": 0, "Fast Food": 1, "Processed": 2},
    "Socioeconomic_Status": {"Low": 0, "Medium": 1, "High": 2},
    "Environmental_Factors": {"Low": 0, "Medium": 1, "High": 2},
    "Region": {"Urban": 0, "Rural": 1},
    "Industrialization_Level": {"Low": 0, "Moderate": 1, "High": 2},
    "Nutrition_Access": {"Good": 0, "Average": 1, "Excellent": 2}
}

@app.route('/')
def index():
    """Render the input form."""
    return render_template('index.html', features=features, categorical_mappings=categorical_mappings)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    try:
        input_data = []
        for feature in features:
            value = request.form.get(feature)
            # Encode categorical features using the predefined mapping
            if feature in categorical_mappings:
                value = categorical_mappings[feature].get(value)
            else:
                value = float(value)  # Convert numerical inputs to float
            input_data.append(value)

        # Reshape input for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Make prediction and round to the nearest integer
        predicted_age = int(round(model.predict(input_array)[0]))

        # Classify as Early or Late
        if predicted_age < 13:
            classification = "Early"
        else:
            classification = "Late"

        return render_template('result.html', result=f"The predicted Menstrual Onset Age is: {predicted_age} years. It is classified as {classification}.")
    
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
