from tensorflow.keras.models import load_model
import joblib

def load_model_and_scaler(fd):
    model = load_model(
        f"models/grumodel_{fd}.h5",
        compile=False   # 🔥 THIS FIXES THE ERROR
    )
    scaler = joblib.load(f"scalers/scaler_{fd}.joblib")
    return model, scaler
