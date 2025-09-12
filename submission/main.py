import os

from submission.predictor import ImmuneStatePredictor
from submission.utils import save_tsv


def main(train_dir: str, test_dir: str, out_dir: str) -> None:
    predictor = ImmuneStatePredictor() # instantiate with any parameters as defined by you in the class
    print(f"Fitting model on ` {train_dir} `...")
    predictor.fit(train_dir)
    print(f"Predicting on ` {test_dir} `...")
    preds = predictor.predict_proba(test_dir)
    if preds is None or preds.empty:
        print("No predictions returned; aborting save.")
    else:
        preds_path = os.path.join(out_dir, f"{os.path.basename(test_dir)}_predictions.tsv")
        save_tsv(preds, preds_path)
        print(f"Predictions written to ` {preds_path} `.")
    seqs = predictor.important_sequences_
    if seqs is not None and not seqs.empty:
        seqs_path = os.path.join(out_dir, f"{os.path.basename(train_dir)}_important_sequences.tsv")
        save_tsv(seqs, seqs_path)
        print(f"Important sequences written to ` {seqs_path} `.")
    else:
        print("No important sequences to save.")
