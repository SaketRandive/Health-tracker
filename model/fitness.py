# models/fitness_model.py
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_fitness_model():
    """
    Train a model to predict calories burned based on steps
    with more realistic data and better handling
    """
    # Enhanced dataset with more data points for better accuracy
    X = np.array([
        [1000], [2000], [3000], [4000], [5000],
        [6000], [7500], [8000], [9000], [10000],
        [12000], [15000], [17500], [20000], [25000]
    ])  # Steps
    
    y = np.array([
        50, 85, 125, 160, 200, 
        240, 300, 325, 365, 400, 
        480, 600, 700, 800, 1000
    ])  # Calories burned
    
    # Create and train model
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate model accuracy
    r2_score = model.score(X, y)
    print(f"Model trained with accuracy (R² score): {r2_score:.4f}")
    
    # Save model
    model_path = os.path.join(os.path.dirname(__file__), "fitness_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return model

def predict_calories(steps):
    """
    Predict calories burned based on number of steps
    
    Args:
        steps (int): Number of steps taken
        
    Returns:
        float: Estimated calories burned
    """
    model_path = os.path.join(os.path.dirname(__file__), "fitness_model.pkl")
    
    # Check if model exists, if not train it
    if not os.path.exists(model_path):
        model = train_fitness_model()
    else:
        # Load existing model
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    
    # Validate input
    if steps < 0:
        raise ValueError("Steps cannot be negative")
    
    # Predict calories
    calories = model.predict([[steps]])[0]
    
    # Round to 1 decimal place for cleaner output
    return round(calories, 1)

def get_activity_level(steps):
    """
    Return activity level classification based on steps
    
    Args:
        steps (int): Number of steps
        
    Returns:
        str: Activity level classification
    """
    if steps < 5000:
        return "Sedentary"
    elif steps < 7500:
        return "Lightly Active"
    elif steps < 10000:
        return "Moderately Active"
    elif steps < 12500:
        return "Active"
    else:
        return "Very Active"

def get_steps_goal(current_steps, goal="improve"):
    """
    Suggest a step goal based on current activity level
    
    Args:
        current_steps (int): Current daily steps
        goal (str): Goal type ("maintain", "improve", or "challenge")
        
    Returns:
        int: Recommended steps goal
    """
    if goal == "maintain":
        return max(current_steps, 7500)
    elif goal == "improve":
        return max(current_steps * 1.2, 10000)
    elif goal == "challenge":
        return max(current_steps * 1.5, 12000)
    else:
        return 10000  # Default goal

# Initialize model when module is imported
if __name__ == "__main__":
    train_fitness_model()