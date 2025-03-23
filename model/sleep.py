import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Sample dataset (sleep hours vs. quality)
X = np.array([[4], [5], [6], [7], [8]])  # Hours of sleep
y = np.array([0, 0, 1, 1, 1])  # Poor (0) or Good (1)

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Save Model
with open("sleep_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Function to predict sleep quality
def predict_sleep_quality(hours):
    with open("sleep_model.pkl", "rb") as f:
        model = pickle.load(f)
    return "Good Sleep" if model.predict([[hours]])[0] == 1 else "Poor Sleep"

# Additional function to provide sleep recommendations
def get_sleep_recommendation(hours):
    if hours < 5:
        return "You should sleep more! Aim for at least 7 hours."
    elif 5 <= hours < 7:
        return "Your sleep is decent, but try to get a bit more rest."
    else:
        return "Great job! You're getting enough sleep."

# Function to visualize sleep quality data
def plot_sleep_quality():
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, model.predict(X), color='red', linestyle='dashed', label='Regression Line')
    plt.xlabel("Hours of Sleep")
    plt.ylabel("Sleep Quality (0 = Poor, 1 = Good)")
    plt.title("Hours of Sleep vs. Sleep Quality")
    plt.legend()
    plt.show()




# import google.generativeai as genai
# import os
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# def predict_sleep_quality(hours):
#     """ Predict sleep quality based on hours slept using Gemini AI """
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = f"Evaluate sleep quality for {hours} hours of sleep."
#     response = model.generate_content(prompt)
#     return response.text.strip()
