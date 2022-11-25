import pandas as pd
import numpy as np

if __name__ == "__main__":
    np.random.seed(1)

    df = pd.read_excel("data/TCGA_MEASUREMENTS.xlsx")

    # Exclude columns that are manually labeled by experts
    cols = [
        "ID",
        "ADI",
        "BACK",
        "DEB",
        "LYM",
        "MUC",
        "MUS",
        "NORM",
        "STR",
        "TUM",
        "years_to_birth",
        "vital_status",
        "gender",
        "days_to_event",
    ]
    df = df[cols]

    n = df.shape[0]
    perm = np.random.permutation(n)

    df["gender"] = df["gender"].replace({"male": 0, "female": 1})
    df["years_to_birth"] = df["years_to_birth"].fillna(-1)

    X = df.loc[:, ~df.columns.isin(["days_to_event", "vital_status", "ID"])]
    y = df[["vital_status", "days_to_event"]]
    y["vital_status"] = df[["vital_status"]].astype(bool)

    X_train = X[X.index.isin(perm[: int(0.8 * n)])]
    X_test = X[~X.index.isin(perm[: int(0.8 * n)])]
    y_train = y[y.index.isin(perm[: int(0.8 * n)])]
    y_test = y[~y.index.isin(perm[: int(0.8 * n)])]

    X_train.to_pickle("data/TCGA_X_train.pkl")
    y_train.to_pickle("data/TCGA_y_train.pkl")
    X_test.to_pickle("data/TCGA_X_test.pkl")
    y_test.to_pickle("data/TCGA_y_test.pkl")
