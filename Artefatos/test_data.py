import pytest
import wandb
import pandas as pd

# This is global so all tests are collected under the same run
run = wandb.init(project="nlp_bolsa", job_type="data_checks")

@pytest.fixture(scope="session")
def data():

    local_path = run.use_artifact("nlp_bolsa/preprocessed_data.csv:latest").file()
    df = pd.read_csv(local_path)

    return df

def test_data_length(data):
    """
    Nosso teste possui dados suficientes para continuar
    """
    assert len(data) > 1000


def test_number_of_columns(data):
    """
    Nós teste possui o número de colunas para continuar
    """
    assert data.shape[1] == 2

def test_column_presence_and_type(data):

    required_columns = {
        "assunto_cnj": pd.api.types.is_object_dtype,
        "conteudo_sentenca": pd.api.types.is_object_dtype
    }

    # Verifica a presença da coluna
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    # Check that only the known classes are present
    known_classes = [
        "Direito Tributário",
        "Direito Civil",
        "Direito Previdenciário",
        "Direito Administrativo e outras matérias do Direito Público",
        "Direito do Consumidor",
        "Direito Processual Civil e do Trabalho",
        "Direito do Trabalho"
    ]

    assert data["assunto_cnj"].isin(known_classes).all()