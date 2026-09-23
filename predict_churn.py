from pathlib import Path

import click
import numpy as np
import pandas as pd
from pycaret.classification import load_model, predict_model

BASE_DIR = Path(__file__).resolve().parent


class ChurnPredictor:
    """
    Loads the saved PyCaret churn model and scores new customer data.
    """

    FEATURES = [
        "tenure",
        "PhoneService",
        "Contract",
        "MonthlyCharges",
        "TotalCharges",
        "PaymentMethod_Credit card (automatic)",
        "PaymentMethod_Electronic check",
        "PaymentMethod_Mailed check",
        "avg_monthly",
    ]
    CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    PAYMENT_CODES = {
        0: "Credit card (automatic)",
        1: "Mailed check",
        2: "Electronic check",
        3: "Bank transfer (automatic)",
    }

    def __init__(self, model_name="churn_lr_model", train_probs="train_churn_probs.csv"):
        self.model = load_model(str(BASE_DIR / model_name), verbose=False)
        self.train_probs = pd.read_csv(BASE_DIR / train_probs).iloc[:, 0].to_numpy()

    def load_data(self, filepath):
        """
        Loads customer data into a DataFrame indexed by customerID.
        """
        return pd.read_csv(filepath, index_col="customerID")

    def preprocess(self, df):
        """
        Applies the Week 2 preprocessing so new data matches the training features.
        Handles both the unmodified file (text categories) and the pre-encoded file.
        """
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

        if df["PhoneService"].dtype == object:
            df["PhoneService"] = (df["PhoneService"] == "Yes").astype(int)
        if df["Contract"].dtype == object:
            df["Contract"] = df["Contract"].map(self.CONTRACT_MAP)
        if pd.api.types.is_numeric_dtype(df["PaymentMethod"]):
            df["PaymentMethod"] = df["PaymentMethod"].map(self.PAYMENT_CODES)

        df["avg_monthly"] = np.where(df["tenure"] > 0, df["TotalCharges"] / df["tenure"], 0)
        df = pd.get_dummies(df, columns=["PaymentMethod"])
        return df.reindex(columns=self.FEATURES, fill_value=False)

    def predict(self, df):
        """
        Returns churn probability, its percentile in the training distribution,
        and the predicted label for each row in the DataFrame.
        """
        X = self.preprocess(df)
        scored = predict_model(self.model, data=X, raw_score=True)
        probs = scored["prediction_score_1"].to_numpy()
        return pd.DataFrame(
            {
                "churn_probability": probs.round(3),
                "percentile": [round((self.train_probs < p).mean() * 100, 1) for p in probs],
                "prediction": scored["prediction_label"].map({1: "Churn", 0: "No churn"}).to_numpy(),
            },
            index=df.index,
        )


@click.command()
@click.option(
    "--file",
    "filepath",
    default=str(BASE_DIR / "new_churn_data_unmodified.csv"),
    show_default=True,
    help="CSV of customers to score.",
)
def main(filepath):
    predictor = ChurnPredictor()
    df = predictor.load_data(filepath)
    print("predictions:")
    print(predictor.predict(df))


if __name__ == "__main__":
    main()
