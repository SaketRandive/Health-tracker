# import numpy as np
# import pickle
# import os
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.pipeline import Pipeline

# class DiseaseRiskModel:
#     def __init__(self, model_path="disease_model.pkl"):
#         """Initialize the disease risk prediction model."""
#         self.model_path = model_path
#         self.model = None
#         self.scaler = None
#         self.feature_names = ["Age", "BMI"]
#         self.classes = ["Low Risk", "High Risk"]
        
#     def train(self, X, y, cv=3, random_state=42):
#         """Train the model on patient data.
        
#         Args:
#             X: Features array (Age, BMI)
#             y: Labels array (0=Low Risk, 1=High Risk)
#             cv: Number of cross-validation folds
#             random_state: Random seed for reproducibility
        
#         Returns:
#             Dictionary containing model metrics
#         """
#         # Create pipeline with scaling
#         self.scaler = StandardScaler()
#         pipeline = Pipeline([
#             ('scaler', self.scaler),
#             ('classifier', LogisticRegression(random_state=random_state, max_iter=1000))
#         ])
        
#         # Cross-validation to estimate model performance
#         cv_scores = cross_val_score(
#             pipeline, X, y, 
#             cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
#             scoring='accuracy'
#         )
        
#         # Fit the final model on all data
#         pipeline.fit(X, y)
#         self.model = pipeline
        
#         # Extract coefficients from the model
#         coefficients = self.model.named_steps['classifier'].coef_[0]
#         intercept = self.model.named_steps['classifier'].intercept_[0]
        
#         # Create a decision boundary function
#         def decision_boundary(x1):
#             # For Logistic Regression, decision boundary is: w1*x1 + w2*x2 + b = 0
#             # Solving for x2: x2 = -(w1*x1 + b) / w2
#             return -(coefficients[0] * x1 + intercept) / coefficients[1]
        
#         # Save the model
#         self.save_model()
        
#         return {
#             "cv_accuracy_mean": np.mean(cv_scores),
#             "cv_accuracy_std": np.std(cv_scores),
#             "coefficients": coefficients,
#             "intercept": intercept,
#             "decision_boundary": decision_boundary,
#             "feature_importance": {
#                 name: abs(coef) for name, coef in zip(self.feature_names, coefficients)
#             }
#         }
    
#     def predict(self, age, bmi):
#         """Predict disease risk based on age and BMI.
        
#         Args:
#             age: Patient's age in years
#             bmi: Patient's BMI
            
#         Returns:
#             Dict containing prediction results and probability
#         """
#         if self.model is None:
#             self.load_model()
            
#         # Prepare input features
#         features = np.array([[age, bmi]])
        
#         # Get prediction and probability
#         prediction = self.model.predict(features)[0]
#         probabilities = self.model.predict_proba(features)[0]
        
#         # Extract risk category and probability
#         risk_category = self.classes[prediction]
#         risk_probability = probabilities[prediction]
        
#         # Calculate recommendation threshold based on probability
#         recommendation = self._get_recommendation(risk_category, risk_probability)
        
#         return {
#             "age": age,
#             "bmi": bmi,
#             "risk_category": risk_category,
#             "risk_probability": risk_probability,
#             "high_risk_probability": probabilities[1],
#             "recommendation": recommendation
#         }
    
#     def _get_recommendation(self, risk_category, probability):
#         """Generate recommendation based on risk category and probability."""
#         if risk_category == "High Risk":
#             if probability > 0.8:
#                 return "Immediate medical consultation recommended"
#             elif probability > 0.6:
#                 return "Schedule medical checkup within 30 days"
#             else:
#                 return "Regular monitoring advised"
#         else:
#             if probability > 0.8:
#                 return "Continue healthy lifestyle"
#             else:
#                 return "Consider preventative health measures"
    
#     def batch_predict(self, patient_data):
#         """Predict risk for multiple patients.
        
#         Args:
#             patient_data: List of (age, bmi) tuples
            
#         Returns:
#             List of prediction dictionaries
#         """
#         return [self.predict(age, bmi) for age, bmi in patient_data]
    
#     def visualize(self, save_path=None, plot_data=None):
#         """Visualize the disease risk model with decision boundary.
        
#         Args:
#             save_path: Optional path to save the visualization
#             plot_data: Optional data to plot (X, y) - if None, uses training data
            
#         Returns:
#             Figure object if successful, None otherwise
#         """
#         if self.model is None:
#             self.load_model()
            
#         try:
#             # Create figure
#             fig, ax = plt.subplots(figsize=(10, 8))
            
#             # Create mesh grid for decision boundary
#             x_min, x_max = 20, 80  # Age range
#             y_min, y_max = 15, 45  # BMI range
#             xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
#                                  np.linspace(y_min, y_max, 100))
            
#             # Get predictions for mesh grid
#             Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
#             Z = Z.reshape(xx.shape)
            
#             # Plot decision boundary
#             ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
#             ax.contour(xx, yy, Z, colors='k', linewidths=0.5)
            
#             # Plot data points if provided
#             if plot_data is not None:
#                 X_plot, y_plot = plot_data
#                 ax.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, 
#                           edgecolors='k', cmap=plt.cm.coolwarm)
            
#             # Set labels and title
#             ax.set_title('Disease Risk Prediction Model')
#             ax.set_xlabel('Age (years)')
#             ax.set_ylabel('BMI')
#             ax.set_xlim(x_min, x_max)
#             ax.set_ylim(y_min, y_max)
            
#             # Add legend
#             legend_elements = [
#                 plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3777b3', 
#                            label='Low Risk', markersize=10),
#                 plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#bc4749', 
#                            label='High Risk', markersize=10)
#             ]
#             ax.legend(handles=legend_elements, loc='best')
            
#             # Add grid
#             ax.grid(True, alpha=0.3)
            
#             # Save if requested
#             if save_path:
#                 plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
#             return fig
#         except Exception as e:
#             print(f"Visualization error: {e}")
#             return None
    
#     def save_model(self):
#         """Save the trained model to disk."""
#         if self.model is not None:
#             with open(self.model_path, "wb") as f:
#                 pickle.dump(self.model, f)
    
#     def load_model(self):
#         """Load the model from disk."""
#         if os.path.exists(self.model_path):
#             with open(self.model_path, "rb") as f:
#                 self.model = pickle.load(f)
#         else:
#             raise FileNotFoundError(f"Model file not found: {self.model_path}")


# # Example usage
# if __name__ == "__main__":
#     # Sample dataset with more diverse data
#     X = np.array([
#         [25, 22], [28, 24], [30, 27], [35, 26], 
#         [40, 30], [45, 31], [50, 32], [55, 33],
#         [60, 35], [65, 36], [70, 40], [72, 38]
#     ])  # (Age, BMI)
    
#     y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # 0 = Low Risk, 1 = High Risk
    
#     # Create and train model
#     risk_model = DiseaseRiskModel()
#     metrics = risk_model.train(X, y)
    
#     print("Disease risk model trained successfully!")
#     print(f"Cross-validation accuracy: {metrics['cv_accuracy_mean']:.2f} ± {metrics['cv_accuracy_std']:.2f}")
#     print("\nFeature importance:")
#     for feature, importance in sorted(metrics['feature_importance'].items(), 
#                                      key=lambda x: x[1], reverse=True):
#         print(f"- {feature}: {importance:.3f}")
    
#     # Make predictions for sample patients
#     test_patients = [
#         (27, 23),  # Young with normal BMI
#         (45, 32),  # Middle age with high BMI
#         (65, 38),  # Older with high BMI
#     ]
    
#     print("\nSample predictions:")
#     for patient in risk_model.batch_predict(test_patients):
#         print(f"Age: {patient['age']}, BMI: {patient['bmi']}")
#         print(f"Risk category: {patient['risk_category']} (probability: {patient['risk_probability']:.2f})")
#         print(f"Recommendation: {patient['recommendation']}")
#         print("-" * 40)
    
#     # Visualize model with training data
#     risk_model.visualize(save_path="disease_risk_visualization.png", plot_data=(X, y))
#     print("\nVisualization saved to 'disease_risk_visualization.png'")
    
#     # Simplified function for quick predictions (compatible with original code)
#     def predict_disease_risk(age, bmi):
#         model = DiseaseRiskModel()
#         result = model.predict(age, bmi)
#         return result['risk_category']
# File: model/dieases.py
# File: model/dieases.py
import numpy as np
import pickle
import os
import sklearn
from sklearn.linear_model import LogisticRegression

class DiseaseModel:
    def __init__(self, model_path=None):
        """Initialize the disease risk model."""
        if model_path is None:
            # Use a relative path if no path is specified
            self.model_path = os.path.join(os.path.dirname(__file__), "disease_model.pkl")
        else:
            self.model_path = model_path
        self.model = None
    
    def train(self, X, y):
        """Train a new model with the provided data."""
        # Train the model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, y)
        
        # Save the model
        self.save_model()
        
        return {
            "coefficients": self.model.coef_[0].tolist(),
            "intercept": self.model.intercept_[0],
            "classes": self.model.classes_.tolist()
        }
    
    def save_model(self):
        """Save the model to disk."""
        if self.model is not None:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
            
            return True
        return False
    
    def load_model(self):
        """Load the model from disk."""
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            return True
        return False
    
    def predict(self, age, bmi):
        """Predict disease risk based on age and BMI."""
        # Load the model if not already loaded
        if self.model is None:
            if not self.load_model():
                return "Unable to predict - model not found"
        
        # Make prediction
        try:
            prediction = self.model.predict([[age, bmi]])[0]
            return "High Risk" if prediction == 1 else "Low Risk"
        except Exception as e:
            return f"Error during prediction: {str(e)}"


# Create a simple function that matches your original interface
def predict_disease_risk(age, bmi):
    """Simple wrapper function for backwards compatibility."""
    model = DiseaseModel()
    return model.predict(age, bmi)


# This section will run if the file is executed directly
# It will create and save a new model file
if __name__ == "__main__":
    print("Creating and saving disease risk model...")
    
    # Sample training data
    X = np.array([
        [25, 22], [28, 24], [30, 27], [35, 26], 
        [40, 30], [45, 31], [50, 32], [55, 33],
        [60, 35], [65, 36], [70, 40], [72, 38]
    ])  # (Age, BMI)
    
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])  # 0 = Low Risk, 1 = High Risk
    
    # Create, train and save model
    model = DiseaseModel()
    results = model.train(X, y)
    
    print(f"Model saved to: {model.model_path}")
    print(f"Model coefficients: {results['coefficients']}")
    print("Test prediction for age=30, bmi=25:", model.predict(30, 25))
    print("Test prediction for age=60, bmi=35:", model.predict(60, 35))