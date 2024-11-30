import logging

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def data_drift(df_new: pd.DataFrame) -> bool:
    """Detecta data drift entre o dataset de produção e um novo dataset.

    :param df_new: Novo dataset.
    :type df_new: pd.DataFrame
    :return: Se houve data drift.
    :rtype: bool
    """    

    df_prod = pd.read_csv("data/user_behavior_dataset_processed.csv")
    df_prod.drop(["User Behavior Class"], axis=1, inplace=True)
    
    if "User Behavior Class" in df_new.columns:
        df_new.drop(["User Behavior Class"], axis=1, inplace=True)

    df_prod["Set"] = 0
    df_new["Set"] = 1

    df = pd.concat([df_prod, df_new], axis=0).sample(frac=1, random_state=42)

    X = df.drop(["Set"], axis=1)
    y = df["Set"]

    categorical_cols = ["Device Model", "Operating System", "Gender"]
    numerical_cols = [
        "App Usage Time (min/day)",
        "Screen On Time (hours/day)",
        "Battery Drain (mAh/day)",
        "Number of Apps Installed",
        "Data Usage (MB/day)",
        "Age",
    ]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )

    col_transf = make_column_transformer(
        (StandardScaler(), numerical_cols),
        (OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy}")

    conf_mat = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_mat, display_labels=model.classes_)
    disp.plot()

    plt.savefig("results/data_drift_confusion_matrix.png")

    mlflow.log_artifact("results/data_drift_confusion_matrix.png")

    if accuracy > 0.6:
        logging.warning("Data drift detected!")
        return True, accuracy
    
    logging.info("No data drift detected")
    return False, accuracy


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-18s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d %H:%M",
    )
    df_new = pd.read_csv("data/user_behavior_dataset_processed.csv").sample(frac=0.6, random_state=42)
    
    mlflow.set_experiment("Data Drift")
    with mlflow.start_run():
        run_name = "Drift Detection"
        mlflow.set_tag("mlflow.runName", run_name)
        
        drift_detect = data_drift(df_new)
        mlflow.log_metric("data_drift", drift_detect[0])
        mlflow.log_param("accuracy", drift_detect[1])
