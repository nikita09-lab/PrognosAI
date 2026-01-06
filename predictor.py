import numpy as np
from backend.preprocessing import CAP_RUL

def predict_rul(model, X):
    preds = model.predict(X, verbose=0).ravel()
    preds = np.clip(preds, 0, CAP_RUL)
    return preds
