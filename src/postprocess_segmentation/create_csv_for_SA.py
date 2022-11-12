from typing import Literal
import pandas as pd

NUM_TO_CLASS = {
    "0": "ADI",
    "1": "BACK",
    "2": "DEB",
    "3": "LYM",
    "4": "MUC",
    "5": "MUS",
    "6": "NORM",
    "7": "STR",
    "8": "TUM",
}
PATIENT_DATA = "src/postprocess_segmentation/patient_reference_data.csv"


class postProcessProbabilities:
    def __init__(
        self,
        probabilities_file: str,
        mode: Literal["average", "highest_tum"],
        reject_background_treshold: float = 0.9,
    ):
        self.patients = pd.read_csv(PATIENT_DATA)
        self.probabilities = pd.read_csv(probabilities_file)
        self.probabilities.rename(mapper=NUM_TO_CLASS, axis=1, inplace=True)
        self.reject_background_treshold = reject_background_treshold

        if mode == "average":
            self.__average()
        elif mode == "highest_tum":
            self.__highest_tum()

    def __average(self):
        for ind, patient in self.patients.iterrows():
            images = self.probabilities[
                self.probabilities["ID"].str.startswith(patient.ID)
            ].drop("ID", axis=1)
            images = images[images["BACK"] < self.reject_background_treshold]
            if not images.empty:
                final_prob = images.mean()
                for class_type, prob in final_prob.to_dict().items():
                    self.patients.at[ind, class_type] = prob
            else:
                print("Patient not found ", patient.ID)
                self.patients.drop(index=ind, axis=0, inplace=True)

    def __highest_tum(self):
        for ind, patient in self.patients.iterrows():
            images = self.probabilities[
                self.probabilities["ID"].str.startswith(patient.ID)
            ].drop("ID", axis=1)
            images = images[images["BACK"] < self.reject_background_treshold]
            if not images.empty:
                final_prob = images[images["TUM"] == images["TUM"].max()]
                for class_type, prob in final_prob.to_dict("list").items():
                    self.patients.at[ind, class_type] = prob
            else:
                print("Patient not found ", patient.ID)
                self.patients.drop(index=ind, axis=0, inplace=True)

    def save(self, target_file: str):
        self.patients.to_csv(target_file, index=False)


if __name__ == "__main__":
    process = postProcessProbabilities(
        probabilities_file="./TCGA_probabilities_per_image.csv", mode="average"
    )
    process.save("TCGA_SA_data_average.csv")

    process = postProcessProbabilities(
        probabilities_file="./TCGA_probabilities_per_image.csv", mode="highest_tum"
    )
    process.save("TCGA_SA_data_highest_tum.csv")
