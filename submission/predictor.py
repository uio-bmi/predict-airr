import os
import numpy as np
import pandas as pd
from submission.utils import load_data_generator, load_full_dataset, get_repertoire_ids, \
    generate_random_top_sequences_df


class ImmuneStatePredictor:
    """
    A template for predicting immune states from TCR repertoire data.

    Participants should implement the logic for training, prediction, and
    sequence identification within this class.
    """

    def __init__(self, n_jobs: int = 1, device: str = 'cpu', **kwargs):
        """
        Initializes the predictor.

        Args:
            n_jobs (int): Number of CPU cores to use for parallel processing.
            device (str): The device to use for computation (e.g., 'cpu', 'cuda').
            **kwargs: Additional hyperparameters for the model.
        """
        self.n_jobs = n_jobs
        self.device = device
        # --- your code starts here ---
        # Example: Store hyperparameters, the actual model, identified important sequences, etc.
        self.model = None
        self.important_sequences_ = None
        # --- your code ends here ---

    def fit(self, train_dir_path: str):
        """
        Trains the model on the provided training data.

        Args:
            train_dir_path (str): Path to the directory with training TSV files.

        Returns:
            self: The fitted predictor instance.
        """

        # --- your code starts here ---
        # Load the data, prepare suited representations as needed, train your model,
        # and find the top k important sequences that best explain the labels.
        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_train_dataset_df = load_full_dataset(train_dir_path)

        #   Model Training
        #    Example: self.model = SomeClassifier().fit(X_train, y_train)
        self.model = "some trained model"  # Replace with your actual learnt model

        #   Identify important sequences (can be done here or in the dedicated method)
        #    Example:
        self.important_sequences_ = self.identify_associated_sequences(top_k=50000, dataset_name=os.path.basename(train_dir_path))

        # --- your code ends here ---
        print("Training complete.")
        return self

    def predict_proba(self, test_dir_path: str) -> pd.DataFrame:
        """
        Predicts probabilities for the test data.

        Args:
            test_dir_path (str): Path to the directory with test TSV files.

        Returns:
            pd.DataFrame: A DataFrame with 'sample_id' and 'probability' columns.
        """
        print(f"Making predictions for data in {test_dir_path}...")
        if self.model is None:
            raise RuntimeError("The model has not been fitted yet. Please call `fit` first.")

        # --- your code starts here ---

        # Example: Load the data. One possibility could be to use the provided utility function as shown below.

        # full_test_dataset_df = load_full_dataset(test_dir_path)
        repertoire_ids = get_repertoire_ids(test_dir_path)  # Replace with actual repertoire IDs from the test data

        # Prediction
        #    Example:
        # draw random probabilities for demonstration purposes

        probabilities = np.random.rand(len(repertoire_ids)) # Replace with true predicted probabilities from your model

        # --- your code ends here ---

        predictions_df = pd.DataFrame({
            'ID': repertoire_ids,
            'dataset': [os.path.basename(test_dir_path)] * len(repertoire_ids),
            'label_positive_probability': probabilities
        })

        print("Prediction complete.")
        return predictions_df

    def identify_associated_sequences(self, dataset_name: str, top_k: int = 50000) -> pd.DataFrame:
        """
        Identifies the top "k" important sequences (rows) from the training data that best explain the labels.

        Args:
            top_k (int): The number of top sequences to return (based on some scoring mechanism).

        Returns:
            pd.DataFrame: A DataFrame with 'ID', 'dataset', junction_aa, v_call, j_call columns.
        """

        # --- your code starts here ---
        # Return the top k sequences, sorted based on some form of importance score.
        # Example:
        # all_sequences_scored = self._score_all_sequences()
        all_sequences_scored = generate_random_top_sequences_df(n_seq=top_k)  # Replace with your way of identifying top k sequences
        top_sequences_df = all_sequences_scored.nlargest(top_k, 'importance_score')
        top_sequences_df = top_sequences_df[['junction_aa', 'v_call', 'j_call']]
        top_sequences_df['dataset'] = dataset_name
        top_sequences_df['ID'] = range(1, len(top_sequences_df)+1)
        top_sequences_df['ID'] = top_sequences_df['dataset'] + '_seq_top_' + top_sequences_df['ID'].astype(str)
        top_sequences_df = top_sequences_df[['ID', 'dataset', 'junction_aa', 'v_call', 'j_call']]

        # --- your code ends here ---
        return top_sequences_df
