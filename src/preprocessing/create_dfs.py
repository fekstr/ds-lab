import os
import pickle

import pandas as pd
import numpy as np

TABULAR_PATH = "data/TCGA_MEASUREMENTS.xlsx"
SLIDES_PATH = "data/TCGA_processed"

if __name__ == "__main__":
    np.random.seed(1)

    df = pd.read_excel(TABULAR_PATH, index_col="ID")

    # Exclude columns that are manually labeled by experts
    cols = [
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

    patient_ids = df.index.to_numpy()
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
    df = df[df.index.isin(patient_slide_map.keys())]
    patient_slide_map = pd.DataFrame(
        [[v] for v in patient_slide_map.values()],
        index=patient_slide_map.keys(),
        columns=["slide_paths"],
    )
    df = df.join(patient_slide_map, how="inner")

    df["gender"] = df["gender"].replace({"male": 0, "female": 1})
    df["years_to_birth"] = df["years_to_birth"].fillna(-1)
    df = df.astype({"vital_status": bool})

    n = df.shape[0]
    perm = np.random.permutation(n)
    i_train = perm[: int(0.8 * n)]
    i_test = np.setdiff1d(perm, i_train)

    df_train = df.iloc[i_train]
    df_test = df.iloc[i_test]

    df_train.to_pickle("data/TCGA_train.pkl")
    df_test.to_pickle("data/TCGA_test.pkl")
