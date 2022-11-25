import pandas as pd

from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis


if __name__ == "__main__":
    X_train = pd.read_pickle("data/TCGA_X_train.pkl")
    y_train = pd.read_pickle("data/TCGA_y_train.pkl")
    X_test = pd.read_pickle("data/TCGA_X_test.pkl")
    y_test = pd.read_pickle("data/TCGA_y_test.pkl")

    y_train = y_train.set_index("vital_status").to_records(
        index_dtypes={"vital_status": "?"},
        column_dtypes={"days_to_event": "<f8"},
    )

    est = CoxPHSurvivalAnalysis()
    est.fit(X_train, y_train)
    risk_preds = est.predict(X_test)

    (
        c_index,
        n_concordant,
        n_discordant,
        n_tied_risk,
        n_tied_time,
    ) = concordance_index_censored(
        y_test["vital_status"], y_test["days_to_event"], risk_preds
    )

    print("C-index:", c_index)
    print("Concordant pairs:", n_concordant)
    print("Discordant pairs:", n_discordant)
    print("Tied risk pairs:", n_tied_risk)
    print("Tied time pairs:", n_tied_time)
