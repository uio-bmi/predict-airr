## Adaptive Immune Profiling Challenge 2025 - Code Template Repository

This repository provides a code template to encourage a unified way of running the models of different participants/teams from the AIRR-ML-25 community challenge: https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025

As described in the [official competition page](https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025), to win the prize money, a prerequisite is that the code has to be made **open-source**. In addition, the top 10 submissions/teams will be invited to become co-authors in a scientific paper that involves further stress-testing of their models in a subsequent phase with many other datasets outside Kaggle platform. To enable such further analyses and re-use of the models by the community, **we strongly encourage** the participants to adhere to a [code template](https://github.com/uio-bmi/predict-airr) that we provide through this repository that enables a uniform interface of running models: https://github.com/uio-bmi/predict-airr

Ideally, all the methods can be run in a unified way, at a later stage, e.g.,

`python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir --out_dir /path/to/output_dir --n_jobs 4 --device cpu`

OR 

```
train_datasets_dir = "/path/to/train_datasets"
test_datasets_dir = "/path/to/test_datasets"
results_dir = "/path/to/results"

train_test_dataset_pairs = get_dataset_pairs(train_datasets_dir, test_datasets_dir) # provided utility function

for train_dir, test_dirs in train_test_dataset_pairs:
    main(train_dir=train_dir, test_dirs=test_dirs, out_dir=results_dir, n_jobs=4, device="cpu")
```

This requires that participants/teams adhere to the code template in `ImmuneStatePredictor` class provided in `predictor.py` by filling in their implementations within the placeholders and replacing any example code lines with actual code that makes sense. 

It will also be important for the participants/teams to provide the exact requirements/dependencies to be able to containerize and run their code. If the participants/teams fork the provided repository and make their changes, it has to be remembered to also replace the dependencies in `requirements.txt` with their dependencies and exact versions.

## Note to those developing code on Kaggle Notebooks
Those participants that make use of Kaggle resources and Kaggle notebooks to develop and run their code are also **strongly encouraged** to copy the code template, particularly the `ImmuneStatePredictor` class and any utility functions from the provided code template repository and adhere to the code template to enable unified way of running different methods at a later stage. We also provided one example notebook on the Kaggle platform, where we copied the code template.

## Generating the final submissions.csv file

As shown in the command line above, we assume that the implementations will adhere to a uniform interface. Regardless of whether one has a loop around the `main` function or not, we assume the output directory will contain multiple `*_test_predictions.tsv` files, one per each training dataset, and multiple `*_important_sequences.tsv` files, one per each training dataset. We provide one utility function named `concatenate_output_files` under `utils.py` that can be used to generate the final `submissions.csv` file from these individual prediction files.

```
results_dir = "/path/to/results"
concatenate_output_files(out_dir=results_dir) # provided utility function
```