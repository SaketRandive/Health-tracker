# backend.py
import os
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
from flask import Flask, request, jsonify
from model.fitness import predict_calories
from model.diet import get_diet_recommendation
from model.sleep import predict_sleep_quality
from model.dieases import predict_disease_risk

app = Flask(__name__)

@app.route('/predict_calories', methods=['POST'])
def calories():
    data = request.json
    steps = data['steps']
    return jsonify({'calories_burned': predict_calories(steps)})

@app.route('/diet', methods=['GET'])
def diet():
    goal = request.args.get('goal', 'balanced')
    return jsonify({'diet_plan': get_diet_recommendation(goal)})

@app.route('/sleep', methods=['POST'])
def sleep():
    data = request.json
    hours = data['hours']
    return jsonify({'sleep_quality': predict_sleep_quality(hours)})

@app.route('/disease', methods=['POST'])
def disease():
    data = request.json
    age, bmi = data['age'], data['bmi']
    return jsonify({'disease_risk': predict_disease_risk(age, bmi)})

if __name__ == '__main__':
    app.run(debug=True)
