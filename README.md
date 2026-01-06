# forest_cover_prediction

## 📌 Project Overview
This project predicts the **type of forest cover** for a given land area using **cartographic and environmental features**.  
It uses a **Random Forest Classifier** trained on real forest data from the **Roosevelt National Forest (Colorado, USA)** and is deployed as a **user-friendly Streamlit web application**.

The app is designed to be **practical and feasible**, requiring only a few meaningful inputs instead of dozens of raw metrics.

---

## 🎯 Problem Statement
Given environmental and geographical features of a **30m × 30m land patch**, predict the **forest cover type**:

| Label | Forest Cover Type |
|-----|------------------|
| 1 | Spruce / Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood / Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## 🧠 Solution Approach
- Data preprocessing and scaling
- Feature selection and engineering
- Model training using **Random Forest**
- Evaluation using accuracy and classification report
- Deployment using **Streamlit**
- Intelligent auto-filling of non-user-facing features using dataset statistics

---

## 🚀 Key Features
✅ End-to-end ML pipeline  
✅ User-friendly web interface  
✅ No need to input all 55 features  
✅ Feature mismatch-safe deployment  
✅ Reusable trained model  
✅ Industry-style project structure  

---

## 🗂️ Project Structure

forest_cover_prediction/
│
├── app.py # Streamlit web app

├── main.py # Model training & saving

├── requirements.txt

│

├── data/

│ └── train.csv

│

├── models/

│ ├── forest_model.pkl

│ ├── scaler.pkl

│ ├── feature_names.pkl

│ └── feature_means.pkl

│

├── src/

│ ├── init.py

│ ├── data_preprocessing.py

│ ├── train_model.py

│ └── evaluate_model.py

│

└── venv/


---

## 📊 Model Performance
- **Accuracy:** ~87.5%
- Strong and balanced performance across all 7 forest cover classes
- Random Forest chosen for robustness and interpretability

---

## 🖥️ Web Application (Streamlit)
The app asks only for **key environmental inputs** such as:
- Elevation
- Slope
- Aspect
- Distance to water
- Distance to roads
- Distance to fire points
- Wilderness area

All other features are automatically filled using **statistical means** from the dataset.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/forest-cover-prediction.git
cd forest-cover-prediction
```
### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```
### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Train the model
```
python main.py
```
### 5️⃣ Run the web app
```
streamlit run app.py
```

<img width="1690" height="1023" alt="Screenshot 2026-01-06 195432" src="https://github.com/user-attachments/assets/23fbd020-66f2-4a01-8e59-6937be9aef37" />

🧪 Technologies Used

Python
Pandas, NumPy
Scikit-learn
Joblib
Streamlit
VS Code

📈 Future Enhancements

Prediction confidence visualization
Feature importance graphs
Map-based location selection
Online deployment (Streamlit Cloud)
Model comparison (XGBoost, LightGBM)

🎓 Learning Outcomes

Practical machine learning workflow
Feature engineering & scaling
Model persistence and reuse
ML-to-web deployment
Debugging real-world ML issues
