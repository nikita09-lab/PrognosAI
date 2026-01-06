import numpy as np

SEQ_LEN = 50
CAP_RUL = 125

def preprocess_test(df, scaler, sensor_cols):
    df = df.copy()
    df[sensor_cols] = scaler.transform(df[sensor_cols])
    return df

def build_test_sequences(df, sensor_cols):
    X = []
    units = sorted(df["unit"].unique())

    for u in units:
        tmp = df[df["unit"] == u].sort_values("cycle")
        arr = tmp[sensor_cols].values

        if arr.shape[0] >= SEQ_LEN:
            X.append(arr[-SEQ_LEN:])
        else:
            pad = np.tile(arr[0], (SEQ_LEN - arr.shape[0], 1))
            X.append(np.vstack([pad, arr]))

    return np.array(X), units
