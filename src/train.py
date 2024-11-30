import json
import logging

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def configure_logger() -> None:
    """Configura o logger para salvar os logs em um arquivo.
    """    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)-18s %(name)-8s %(levelname)-8s %(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename="logs/model_training.log",
        filemode="a",
    )

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega o dataset separando as features da variável alvo.

    :return: Um dataframe com as features e um dataframe com a variável alvo.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """    
    logging.info("Loading data...")
    df = pd.read_csv("data/user_behavior_dataset_processed.csv")

    X = df.drop(["User Behavior Class"], axis=1)
    y = df["User Behavior Class"]

    logging.info("Data loaded successfully.")
    return X, y


def split_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide o dataset em treino e teste.

    :param X: Dataframe com as features.
    :type X: pd.DataFrame
    :param y: Dataframe com a target.
    :type y: pd.DataFrame
    :return: Dataframes de treino e teste para treino e teste.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """    
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logging.info("Data split successfully.")
    return X_train, X_test, y_train, y_test


def train(X_train: pd.DataFrame, y_train: pd.DataFrame, register_model: bool) -> Pipeline:
    """Treina um modelo de classificação RandomForest.

    :param X_train: Dataframe com as features de treino.
    :type X_train: pd.DataFrame
    :param y_train: Dataframe com a target de treino.
    :type y_train: pd.DataFrame
    :param register_model: Se True, registra o modelo no MLflow.
    :type register_model: bool
    :return: Modelo treinado.
    :rtype: Pipeline
    """    
    logging.info("Training model...")

    categorical_cols = ["Device Model", "Operating System", "Gender"]
    numerical_cols = [
        "App Usage Time (min/day)",
        "Screen On Time (hours/day)",
        "Battery Drain (mAh/day)",
        "Number of Apps Installed",
        "Data Usage (MB/day)",
        "Age",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("random_state", 42)

    model.fit(X_train, y_train)

    if register_model:
        X_train = X_train.copy()
        X_train[numerical_cols] = X_train[numerical_cols].astype(float)

  
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            "user_behavior_model",
            signature=signature,
            registered_model_name="user_behavior_model",
            input_example=X_train.head(5),
        )
    logging.info("Model trained successfully.")
    return model


def export_metrics(y_test: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    """Exporta as métricas de avaliação do modelo.

    :param y_test: Dataframe com a target de teste.
    :type y_test: pd.DataFrame
    :param y_pred: Dataframe com as predições do modelo.
    :type y_pred: pd.DataFrame
    """    
    logging.info("Exporting metrics...")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    logging.info("Metrics exported successfully.")


def export_confusion_matrix(model: Pipeline, y_test: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    """Exporta a matriz de confusão do modelo.

    :param model: Modelo treinado.
    :type model: Pipeline
    :param y_test: Dataframe com a target de teste.
    :type y_test: pd.DataFrame
    :param y_pred: Dataframe com as predições do modelo.
    :type y_pred: pd.DataFrame
    """    
    logging.info("Exporting confusion matrix...")
    conf_matrix = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(conf_matrix, index=model.classes_, columns=model.classes_)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig("results/confusion_matrix.png")
    
    mlflow.log_artifact("results/confusion_matrix.png")
    logging.info("Confusion matrix exported successfully.")


def export_classification_report(y_test: pd.DataFrame, y_pred: pd.DataFrame) -> None:
    """Exporta o relatório de classificação do modelo.

    :param y_test: Dataframe com a target de teste.
    :type y_test: pd.DataFrame
    :param y_pred: Dataframe com as predições do modelo.
    :type y_pred: pd.DataFrame
    """    
    logging.info("Exporting classification report...")
    report = classification_report(y_test, y_pred, output_dict=True)

    with open("results/classification_report.json", "w") as file:
        json.dump(report, file)

    mlflow.log_artifact("results/classification_report.json")

    logging.info("Classification report exported successfully.")

def main(register_model: bool)->None: 
    """Função principal que carrega os dados, treina o modelo, avalia o modelo e exporta as métricas.

    :param register_model: Se True, registra o modelo no MLflow.
    :type register_model: bool
    """           
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train(X_train, y_train, register_model)
    y_pred = model.predict(X_test)

    export_metrics(y_test, y_pred)
    export_confusion_matrix(model, y_test, y_pred)
    export_classification_report(y_test, y_pred)

    logging.info("Model training completed.")


if __name__ == "__main__":
    configure_logger()
    register_model = True
    
    mlflow.set_experiment("User Behavior Classification")
    with mlflow.start_run():
        run_name = "User Behavior Classification Model Training"
        mlflow.set_tag("mlflow.runName", run_name)
        
        main(register_model=register_model)
