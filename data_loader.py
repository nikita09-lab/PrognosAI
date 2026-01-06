import pandas as pd

def load_test_data(fd):
    path = f"data/test_{fd}.txt"
    cols = (
        ["unit", "cycle"] +
        [f"op_setting_{i}" for i in range(1, 4)] +
        [f"s{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.iloc[:, :-2]
    df.columns = cols
    return df
