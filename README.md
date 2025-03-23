# AI-Powered Health Tracker

## 📌 Overview
The **AI-Powered Health Tracker** is a web-based application that helps users monitor their health by analyzing fitness, diet, sleep, and disease risk using machine learning and AI. The system provides real-time insights and recommendations for a healthier lifestyle.

## 🚀 Features
- **Fitness Tracking**: Predict calories burned based on steps walked.
- **Diet Planning**: Get personalized diet recommendations.
- **Sleep Analysis**: Assess sleep quality and receive improvement tips.
- **Disease Risk Prediction**: Estimate potential health risks based on age and BMI.

## 🛠️ Technologies Used
- **Backend**: Flask (Python)
- **Frontend**: Streamlit (Python)
- **Machine Learning**: scikit-learn
- **Database**: SQLite

## 🔧 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/health-tracker.git
cd health-tracker
```
### 2️⃣ Create Virtual Environment & Install Dependencies
```bash
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate     # On Windows
pip install -r requirements.txt
```
### 3️⃣ Run the Backend
```bash
python backend.py
```
### 4️⃣ Run the Frontend
```bash
streamlit run app.py
```

## 🎯 Usage
1. **Enter steps walked** to get calorie predictions.
2. **Choose a fitness goal** (weight loss, muscle gain, balanced) for diet recommendations.
3. **Enter sleep hours** to analyze sleep quality.
4. **Enter age and BMI** to assess disease risk.

## 📌 Contributing
We welcome contributions! Feel free to fork the repo, create a new branch, and submit a PR.

## 📜 License
This project is licensed under the MIT License.
