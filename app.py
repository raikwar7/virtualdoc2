from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
with open("models/disease_model (1).pkl", "rb") as file:
    disease_model, le, unique_symp = pickle.load(file)
with open("models/doctor_model (2).pkl", "rb") as f:
    data_score, unique_locations = pickle.load(f)

df = pd.read_csv("dataset/updated_disease_to_specialization.csv")
disease_to_speciality = dict(zip(df['Disease'], df['Specialization']))

def predict_speciality(disease_name):
    return disease_to_speciality.get(disease_name, "Specialist not found")

# Load additional data
dataf1 = pd.read_csv("dataset/symptom_Description.csv")
datap = pd.read_csv("dataset/symptom_precaution (3).csv")

# Strip column names of whitespace
dataf1.columns = dataf1.columns.str.strip()
datap.columns = datap.columns.str.strip()

# Function to convert symptoms to vector
def sym_to_vec(symptoms, symptom_list):
    symptoms_set = set(symptoms.split(","))
    return [1 if symptom in symptoms_set else 0 for symptom in symptom_list]

# Function to predict disease
def predict_disease(input_symptoms):
    input_vector = np.array(sym_to_vec(",".join(input_symptoms).lower(), unique_symp)).reshape(1, -1)
    probabilities = disease_model.predict_proba(input_vector)[0]

    disease_probabilities = {
        le.inverse_transform([i])[0]: round(prob * 100, 2)
        for i, prob in enumerate(probabilities)
    }
    
    # Sort and get the top 7 predicted diseases
    top_7_diseases = sorted(disease_probabilities.items(), key=lambda x: x[1], reverse=True)[:7]
    
    # Debugging output
    print("\nPredicted Disease Probabilities:", disease_probabilities)
    print("\nTop 7 Predictions:", top_7_diseases)

    return top_7_diseases

# Function to get disease description and precautions
def get_precaution_and_symptoms(disease_name):
    result = dataf1[dataf1["Disease"].str.lower() == disease_name.lower()]
    
    if not result.empty:
        # Get disease description safely
        description = result.iloc[0].get("Description", "No description available")
        
        # Get precautions safely
        precautions = datap[datap["Disease"].str.lower() == disease_name.lower()].iloc[:, 1:].values.flatten().tolist()
        precautions = [p for p in precautions if isinstance(p, str)]  # Remove NaN values

        return description, precautions if precautions else ["No Precautions Available"]

    return "No description available", ["No Precautions Available"]

# Home route
@app.route("/")
def home():
    return render_template("index.html", symptoms=unique_symp)

# Predict disease route

@app.route("/predict", methods=["POST"])
def predict():
    symptoms = request.form.getlist("symptoms")

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    top_7_predictions = predict_disease(symptoms)

    if not top_7_predictions:
        return jsonify({"error": "No predictions available"}), 500

    # Extract top 3 predictions
    top_1 = top_7_predictions[0]
    top_2 = top_7_predictions[1]
    top_3 = top_7_predictions[2] if len(top_7_predictions) > 2 else None

    # Start with top 2
    selected_diseases = [top_1, top_2]

    # Check if the 3rd has the same confidence as either of the top 2
    if top_3 and (top_3[1] == top_1[1] or top_3[1] == top_2[1]):
        selected_diseases.append(top_3)

    # Fetch descriptions and precautions for the selected diseases
    disease_details = []
    for disease, confidence in selected_diseases:
        description, precautions = get_precaution_and_symptoms(disease)
        disease_details.append({
            "disease": disease,
            "confidence": confidence,
            "description": description,
            "precautions": precautions
        })

    return jsonify({
        "selected_diseases": disease_details,
        "top_7": top_7_predictions  # Optional: Return full top 7 for debugging
    })

"""def predict():
    symptoms = request.form.getlist("symptoms")

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    top_7_predictions = predict_disease(symptoms)

    if not top_7_predictions:
        return jsonify({"error": "No predictions available"}), 500

    # Get details for the most likely disease
    predicted_disease = top_7_predictions[0][0]
    confidence = top_7_predictions[0][1]
    description, precautions = get_precaution_and_symptoms(predicted_disease)

    return jsonify({
        "predicted_disease": predicted_disease,
        "confidence": confidence,
        "top_7": top_7_predictions,  # ✅ Full list returned
        "description": description,
        "precautions": precautions
    })
""" 
# Recommend doctor route
"""@app.route("/recommend", methods=["POST"])

def recommend():
    disease_name = request.form.get("disease")
    location = request.form.get("location")

    if not disease_name or not location:
        return jsonify({"error": "Missing disease name or location"}), 400
    
    # Debugging Output
    print("\nReceived Disease:", disease_name)
    print("Received Location:", location)

    # Predict doctor specialization
    disease_vector = vectorizer.transform([disease_name])
    predicted_specialty = doctor_model.predict(disease_vector)[0]

    print("Predicted Specialty:", predicted_specialty)

    recommended_doctors = data_score[
        (data_score["Specialization"] == predicted_specialty) &
        (data_score["Location"] == location)
    ].sort_values(by="score", ascending=False).head(5)

    return jsonify({
        "specialization": predicted_specialty,
        "doctors": recommended_doctors.to_dict(orient="records")
    }) """
    
@app.route("/recommend", methods=["POST"])
def recommend():
    disease_name = request.form.get("disease")
    location = request.form.get("location")

    if not disease_name or not location:
        return jsonify({"error": "Missing disease name or location"}), 400

    # Debugging Output
    print("\nReceived Disease:", disease_name)
    print("Received Location:", location)

    # Load preprocessed doctor data and locations
    with open("models/doctor_model (2).pkl ", "rb") as f:
        data_score, unique_locations = pickle.load(f)

    # Predict specialty using simple lookup
    predicted_specialty = predict_speciality(disease_name)
    print("Predicted Specialty:", predicted_specialty)

    # Filter matching doctors
    filtered_doctors = data_score[
        (data_score["Specialization"].str.strip().str.lower() == predicted_specialty.strip().lower()) &
        (data_score["Location"].str.strip().str.lower() == location.strip().lower())
    ].sort_values(by="score", ascending=False).head(5)

    if filtered_doctors.empty:
        return jsonify({
            "specialization": predicted_specialty,
            "doctors": []
        })

    return jsonify({
        "specialization": predicted_specialty,
        "doctors": filtered_doctors.to_dict(orient="records")
    })

@app.route('/recommend.html')
def recommend_page():
    disease = request.args.get('disease', 'Unknown')
    return render_template('recommend.html', disease=disease,locations=unique_locations)    
    
    

if __name__ == "__main__":
    app.run(debug=True)
