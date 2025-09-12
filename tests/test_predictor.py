import pytest
import os
import shutil
import pandas as pd
from submission.predictor import ImmuneStatePredictor


@pytest.fixture
def test_environment():
    """
    A pytest fixture to set up a temporary environment with mock data.
    This runs before each test that uses it and cleans up afterward.
    """
    temp_dir = "temp_pytest_data"
    train_data_dir = os.path.join(temp_dir, "train_data")
    test_data_dir = os.path.join(temp_dir, "test_data")
    out_dir = os.path.join(temp_dir, "output")

    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, 4):
        df = pd.DataFrame({
            'junction_aa': [f'SEQ{j}' for j in range(5)],
            'v_call': ['V1'] * 5, 'j_call': ['J1'] * 5,
        })
        df.to_csv(os.path.join(train_data_dir, f"sample_{i}.tsv"), sep='\t', index=False)

    for i in range(4, 6):
        df = pd.DataFrame({
            'junction_aa': [f'SEQ{j}' for j in range(3)],
            'v_call': ['V2'] * 3, 'j_call': ['J2'] * 3,
        })
        df.to_csv(os.path.join(test_data_dir, f"sample_{i}.tsv"), sep='\t', index=False)

    yield {"train_dir": train_data_dir, "test_dir": test_data_dir, "out_dir": out_dir}

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_initialization():
    """Test that the predictor initializes with default values."""
    predictor = ImmuneStatePredictor()
    assert predictor.model is None
    assert predictor.important_sequences_ is None
    assert predictor.n_jobs == 1


def test_fit_changes_internal_state(test_environment):
    """Test that fit() trains a model and modifies the instance state."""
    predictor = ImmuneStatePredictor()
    assert predictor.model is None, "Model should be None before fitting."

    fit_return = predictor.fit(train_dir_path=test_environment["train_dir"])
    assert isinstance(fit_return, ImmuneStatePredictor), "fit() should return self."

    assert predictor.model is not None, "Model should not be None after fitting."


def test_predict_proba_raises_error_before_fit(test_environment):
    """Test that predict_proba fails if the model hasn't been fitted."""
    predictor = ImmuneStatePredictor()
    with pytest.raises(RuntimeError, match="model has not been fitted yet"):
        predictor.predict_proba(test_dir_path=test_environment["test_dir"])


def test_predict_proba_returns_correct_format_after_fit(test_environment):
    """Test the output format of predict_proba after fitting."""
    predictor = ImmuneStatePredictor()
    predictor.fit(train_dir_path=test_environment["train_dir"])

    predictions_df = predictor.predict_proba(test_dir_path=test_environment["test_dir"])
    assert isinstance(predictions_df, pd.DataFrame), "Output should be a pandas DataFrame."

    expected_cols = ['ID', 'dataset', 'label_positive_probability']
    assert list(predictions_df.columns) == expected_cols

    assert len(predictions_df) == 2, "Output should have one row per test sample."

    assert pd.api.types.is_numeric_dtype(predictions_df['label_positive_probability'])
    assert (predictions_df['label_positive_probability'] >= 0).all()
    assert (predictions_df['label_positive_probability'] <= 1).all()


def test_identify_associated_sequences_returns_correct_format(test_environment):
    """Test the output format of identify_associated_sequences."""
    predictor = ImmuneStatePredictor()
    predictor.fit(train_dir_path=test_environment["train_dir"])

    top_k = 10
    dataset_name = os.path.basename(test_environment["train_dir"])
    top_seq_df = predictor.identify_associated_sequences(top_k=top_k, dataset_name=dataset_name)

    assert isinstance(top_seq_df, pd.DataFrame)

    expected_cols = ['ID', 'dataset', 'junction_aa', 'v_call', 'j_call']
    assert list(top_seq_df.columns) == expected_cols

    assert len(top_seq_df) == top_k, f"Should return {top_k} sequences."

    assert (top_seq_df['dataset'] == dataset_name).all()


