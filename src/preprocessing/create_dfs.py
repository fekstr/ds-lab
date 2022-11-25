import os
import pickle

import pandas as pd
import numpy as np

TABULAR_PATH = "data/TCGA_MEASUREMENTS.xlsx"
SLIDES_PATH = "data/TCGA_processed"

if __name__ == "__main__":
    np.random.seed(1)

    df = pd.read_excel(TABULAR_PATH)

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

    patient_ids = df["ID"].to_numpy()
    slide_filenames = os.listdir(SLIDES_PATH)
    patient_slide_map = {}
    for patient_id in patient_ids:
        matching_slides = [
            name for name in slide_filenames if name.startswith(patient_id)
        ]
        if len(matching_slides) == 0:
            continue
        patient_slide_map[patient_id] = [
            os.path.join(SLIDES_PATH, name) for name in matching_slides
        ]
    df = df[df["ID"].isin(patient_slide_map.keys())].reset_index(drop=True)

    df["gender"] = df["gender"].replace({"male": 0, "female": 1})
    df["years_to_birth"] = df["years_to_birth"].fillna(-1)

    X = df.loc[:, ~df.columns.isin(["days_to_event", "vital_status"])]
    y = df[["ID", "vital_status", "days_to_event"]]
    y = y.astype({'vital_status': bool})

    n = df.shape[0]
    perm = np.random.permutation(n)

    X_train = X[X.index.isin(perm[: int(0.8 * n)])]
    X_test = X[~X.index.isin(perm[: int(0.8 * n)])]
    y_train = y[y.index.isin(perm[: int(0.8 * n)])]
    y_test = y[~y.index.isin(perm[: int(0.8 * n)])]

    X_train.to_pickle("data/TCGA_X_train.pkl")
    y_train.to_pickle("data/TCGA_y_train.pkl")
    X_test.to_pickle("data/TCGA_X_test.pkl")
    y_test.to_pickle("data/TCGA_y_test.pkl")

    with open("data/TCGA_patient_slide_map.pkl", "wb") as f:
        pickle.dump(patient_slide_map, f)
