import pandas as pd
import os
import sys
import numpy as np

def calculate_stroma_score(data_dir):
    df = pd.read_excel(data_dir)

    x = df.to_numpy()
    x[:,-1] = x[:,-1] / 365.35 # convert 'days to event' to 'years to event'
    x[:,11] = x[:,11] / 10 # convert 'years to birth' to 'decades to birth'
    x = np.append(x, np.zeros((500,1)), axis=1) # add column for HD score

    # Calculate allWeights
    classes_col = ["ADI","BACK", "DEB", "LYM", "MUC","MUS", "NORM", "STR","TUM"]
    df["years_to_event"] = df["days_to_event"]/365.25 # convert 'days to event' to 'years to event'
    df["decades_to_birth"] = df["years_to_birth"]/10 # convert 'years to birth' to 'decades to birth'
    y = df[["years_to_event","vital_status"]]

    cph_models = [CoxPHFitter().fit(pd.concat([df[col], y], axis=1), "years_to_event", "vital_status") for col in classes_col]
    allWeights = np.array([float(cph.summary["exp(coef)"]) for cph in cph_models])

    # Calculate allCuts?
    allCuts = np.array([0.00056, 0.00227, 0.03151, 0.00121, 0.01123, 0.02359, 0.06405, 0.00122, 0.99961]) # Youden cuts

    # Calculate stroma score
    scoreIndices = (np.argwhere(allWeights >= 1)).flatten()
    for i in scoreIndices:
        x[:,-1] = x[:,-1] + (x[:,i+1]>=allCuts[i])*allWeights[i] # +1 retrieve column number in x
    medianTrainingSet = np.median(x[:,-1])
    x[:,-1] = (x[:,-1] >= medianTrainingSet)*1
    stroma_score = x[:,-1]

    runs=["All stages", "Stage i", "Stage ii", "Stage iii", "Stage iv"]
    df["stroma_score"] = stroma_score

    selected_columns = ["stroma_score" ,"cleanstage", "gender", "decades_to_birth" ]
    df_mv = df.dropna(subset=["cleanstage", "decades_to_birth"])
    y_mv = df_mv[["years_to_event","vital_status"]]

    mv_cox = CoxPHFitter().fit(pd.concat([df_mv[selected_columns], y_mv], axis=1), "years_to_event", "vital_status", formula = "stroma_score + cleanstage + C(gender) +decades_to_birth")
    # mv_cox.print_summary()

    stroma_HR = mv_cox.hazard_ratios_[3]
    lci = np.exp(mv_cox.confidence_intervals_["95% lower-bound"][3])
    hci = np.exp(mv_cox.confidence_intervals_["95% upper-bound"][3])
    p = mv_cox._compute_p_values()[3]
    print(f"___Deep Stroma Score Hazard Ratio, {runs[0]}___ \n Stroma HR: {stroma_HR}; \n CI: [{lci}, {hci}]; \n p: {p}")

    return stroma_HR, lci, hci, p