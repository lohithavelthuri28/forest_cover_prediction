import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Forest Cover Type Prediction", layout="centered")

st.title("🌲 Forest Cover Type Prediction")
st.write("User-friendly prediction using key environmental inputs")

# Load saved objects
model = joblib.load("models/forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")
feature_means = joblib.load("models/feature_means.pkl")

st.subheader("Enter Basic Land Details")

# --- User Inputs (ONLY IMPORTANT FEATURES) ---
elevation = st.slider("Elevation (meters)", 0, 5000, 2500)
aspect = st.slider("Aspect (degrees)", 0, 360, 180)
slope = st.slider("Slope (degrees)", 0, 60, 15)

hd_hydro = st.slider("Distance to Water (m)", 0, 5000, 500)
hd_road = st.slider("Distance to Road (m)", 0, 10000, 1000)
hd_fire = st.slider("Distance to Fire Points (m)", 0, 10000, 2000)

wilderness = st.selectbox(
    "Wilderness Area",
    ["Rawah", "Neota", "Comanche Peak", "Cache la Poudre"]
)

# --- Build Input Vector ---
input_dict = feature_means.copy()

# Override with user values
input_dict["Elevation"] = elevation
input_dict["Aspect"] = aspect
input_dict["Slope"] = slope
input_dict["Horizontal_Distance_To_Hydrology"] = hd_hydro
input_dict["Horizontal_Distance_To_Roadways"] = hd_road
input_dict["Horizontal_Distance_To_Fire_Points"] = hd_fire

# Wilderness one-hot
for i, area in enumerate([
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4"
]):
    input_dict[area] = 1 if wilderness.endswith(str(i+1)) else 0

# Ensure correct feature order
final_input = [input_dict[f] for f in feature_names]

if st.button("Predict Forest Cover Type"):
    scaled_input = scaler.transform([final_input])
    prediction = model.predict(scaled_input)[0]

    cover_map = {
        1: "🌲 Spruce/Fir",
        2: "🌲 Lodgepole Pine",
        3: "🌲 Ponderosa Pine",
        4: "🌳 Cottonwood/Willow",
        5: "🌳 Aspen",
        6: "🌲 Douglas-fir",
        7: "🌲 Krummholz"
    }

    st.success(f"Predicted Forest Cover Type: **{cover_map[prediction]}**")
