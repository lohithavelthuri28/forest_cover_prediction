import joblib
import numpy as np

def predict_cover_type(input_data, model_path, scaler):
    model = joblib.load(model_path)
    input_data = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_data)
    return prediction[0]
