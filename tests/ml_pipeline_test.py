import pytest
from unittest import mock
from app.train import load_data, preprocess_data, create_pipeline, train_model


# Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

from sklearn.model_selection import train_test_split

# Test data loading
def test_load_data():
    url = "https://kub-bucket-ouss.s3.eu-west-3.amazonaws.com/NY_House_Dataset.csv"
    df = load_data(url)
    assert not df.empty, "Dataframe is empty"

# Test data preprocessing
def test_preprocess_data():
    df = load_data("https://kub-bucket-ouss.s3.eu-west-3.amazonaws.com/NY_House_Dataset.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    assert len(X_train) > 0, "Training data is empty"
    assert len(X_test) > 0, "Test data is empty"

# Test pipeline creation
def test_create_pipeline():
    pipe = create_pipeline()
    assert "standard_scaler" in pipe.named_steps, "Scaler missing in pipeline"
    assert "Random_Forest" in pipe.named_steps, "RandomForest missing in pipeline"

# Test model training (mocking GridSearchCV)
@mock.patch('app.train.GridSearchCV.fit', return_value=None)
def test_train_model(mock_fit):
    pipe = create_pipeline()
    X_train, X_test, y_train, y_test = preprocess_data(load_data("https://kub-bucket-ouss.s3.eu-west-3.amazonaws.com/NY_House_Dataset.csv"))
    param_grid = {"Random_Forest__n_estimators": [50], "Random_Forest__criterion": ["squared_error"]}
    model = train_model(pipe, X_train, y_train, param_grid)
    assert model is not None, "Model training failed"





def test_data_drift():
    """
    Test to detect if there's a significant data drift between
    a reference split and a current split of the dataset.
    """
    # 1. Charger le même dataset qu'au-dessus
    url = "https://kub-bucket-ouss.s3.eu-west-3.amazonaws.com/NY_House_Dataset.csv"
    df = load_data(url)

    # 2. On crée un split 80% (ref) / 20% (current)
    df_ref, df_current = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Définir le mapping de colonnes pour Evidently
    #    Adapte selon les colonnes de ton dataset
    column_mapping = ColumnMapping(
        target="PRICE",  
        numerical_features=["BEDS", "BATH", "PROPERTYSQFT", "LATITUDE", "LONGITUDE"],
        categorical_features=[]
    )

    # 4. Créer et exécuter le rapport Evidently
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(
        reference_data=df_ref,
        current_data=df_current,
        column_mapping=column_mapping
    )

    # 5. Récupérer les résultats sous forme de dictionnaire
    results = data_drift_report.as_dict()

    # 6. Vérifier le statut global (vérification simplifiée)
    #    - 'dataset_drift' = True si Evidently détecte un drift global
   # Récupérer directement la valeur booléenne
    has_drift = results["metrics"][0]["result"]["dataset_drift"]

    # 7. Définir la logique du test : on échoue si un drift est détecté
    assert not has_drift, "Data drift détecté selon Evidently !"