import os
import tempfile
import pytest
import pandas as pd
from submission.utils import validate_dirs_and_files, concatenate_output_files, load_and_encode_kmers, get_dataset_pairs


def create_dir_with_tsv_and_metadata(dir_path, n_tsv=1):
    os.makedirs(dir_path, exist_ok=True)
    # Create .tsv files
    for i in range(n_tsv):
        with open(os.path.join(dir_path, f"file{i}.tsv"), "w") as f:
            f.write("col1\tcol2\nval1\tval2\n")
    # Create metadata.csv
    with open(os.path.join(dir_path, "metadata.csv"), "w") as f:
        f.write("repertoire_id,filename,label_positive\nrep1,file0.tsv,1\n")


def create_dir_with_tsv(dir_path, n_tsv=1):
    os.makedirs(dir_path, exist_ok=True)
    for i in range(n_tsv):
        with open(os.path.join(dir_path, f"file{i}.tsv"), "w") as f:
            f.write("col1\tcol2\nval1\tval2\n")


def test_validate_dirs_and_files_valid():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir = os.path.join(tmp, "test")
        out_dir = os.path.join(tmp, "out")
        create_dir_with_tsv_and_metadata(train_dir)
        create_dir_with_tsv(test_dir)
        validate_dirs_and_files(train_dir, [test_dir], out_dir)
        assert os.path.isdir(out_dir)


def test_missing_train_dir():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train_missing")
        test_dir = os.path.join(tmp, "test")
        out_dir = os.path.join(tmp, "out")
        create_dir_with_tsv(test_dir)
        with pytest.raises(AssertionError):
            validate_dirs_and_files(train_dir, [test_dir], out_dir)


def test_missing_test_dir():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir = os.path.join(tmp, "test_missing")
        out_dir = os.path.join(tmp, "out")
        create_dir_with_tsv_and_metadata(train_dir)
        with pytest.raises(AssertionError):
            validate_dirs_and_files(train_dir, [test_dir], out_dir)


def test_missing_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir = os.path.join(tmp, "test")
        out_dir = os.path.join(tmp, "out")
        create_dir_with_tsv(train_dir)
        create_dir_with_tsv(test_dir)
        with pytest.raises(AssertionError):
            validate_dirs_and_files(train_dir, [test_dir], out_dir)


def test_no_tsv_in_train():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir = os.path.join(tmp, "test")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(train_dir, exist_ok=True)
        with open(os.path.join(train_dir, "metadata.csv"), "w") as f:
            f.write("repertoire_id,filename,label_positive\nrep1,file0.tsv,1\n")
        create_dir_with_tsv(test_dir)
        with pytest.raises(AssertionError):
            validate_dirs_and_files(train_dir, [test_dir], out_dir)


def test_no_tsv_in_test():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir = os.path.join(tmp, "test")
        out_dir = os.path.join(tmp, "out")
        create_dir_with_tsv_and_metadata(train_dir)
        os.makedirs(test_dir, exist_ok=True)
        with pytest.raises(AssertionError):
            validate_dirs_and_files(train_dir, [test_dir], out_dir)


def test_out_dir_no_write_permission():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir = os.path.join(tmp, "test")
        out_dir = os.path.join(tmp, "no_write/out")
        create_dir_with_tsv_and_metadata(train_dir)
        create_dir_with_tsv(test_dir)
        no_write_dir = os.path.join(tmp, "no_write")
        os.makedirs(no_write_dir, exist_ok=True)
        os.chmod(no_write_dir, 0o400)  # Remove write permission
        try:
            with pytest.raises(SystemExit):
                validate_dirs_and_files(train_dir, [test_dir], out_dir)
        finally:
            os.chmod(no_write_dir, 0o700)  # Restore permissions for cleanup


def test_validate_dirs_and_files_multiple_test_dirs():
    """
    Ensure validate_dirs_and_files accepts multiple test directories in the list.
    """
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir1 = os.path.join(tmp, "test1")
        test_dir2 = os.path.join(tmp, "test2")
        out_dir = os.path.join(tmp, "out_multi")
        create_dir_with_tsv_and_metadata(train_dir)
        create_dir_with_tsv(test_dir1)
        create_dir_with_tsv(test_dir2)
        validate_dirs_and_files(train_dir, [test_dir1, test_dir2], out_dir)
        assert os.path.isdir(out_dir)


def test_concatenate_output_files():
    """
    Test that concatenate_output_files correctly finds, concatenates,
    and saves test predictions and important sequences files to submissions.csv.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "output")
        os.makedirs(out_dir, exist_ok=True)

        pred_df1 = pd.DataFrame({
            'ID': ['rep1', 'rep2'],
            'dataset': ['test1', 'test1'],
            'label_positive_probability': [0.8, 0.6],
            'junction_aa': [-999.0, -999.0],
            'v_call': [-999.0, -999.0],
            'j_call': [-999.0, -999.0]
        })
        pred_df1.to_csv(os.path.join(out_dir, 'train1_test_predictions.tsv'), sep='\t', index=False)

        pred_df2 = pd.DataFrame({
            'ID': ['rep3', 'rep4'],
            'dataset': ['test2', 'test2'],
            'label_positive_probability': [0.7, 0.9],
            'junction_aa': [-999.0, -999.0],
            'v_call': [-999.0, -999.0],
            'j_call': [-999.0, -999.0]
        })
        pred_df2.to_csv(os.path.join(out_dir, 'train2_test_predictions.tsv'), sep='\t', index=False)

        seq_df1 = pd.DataFrame({
            'ID': ['train1_seq_top_1', 'train1_seq_top_2'],
            'dataset': ['train1', 'train1'],
            'label_positive_probability': [-999.0, -999.0],
            'junction_aa': ['CASSLEETQYF', 'CASSLDPNQPQHF'],
            'v_call': ['TRBV20-1', 'TRBV20-1'],
            'j_call': ['TRBJ2-7', 'TRBJ2-7']
        })
        seq_df1.to_csv(os.path.join(out_dir, 'train1_important_sequences.tsv'), sep='\t', index=False)

        concatenate_output_files(out_dir)

        submissions_path = os.path.join(out_dir, 'submissions.csv')
        assert os.path.exists(submissions_path), "submissions.csv was not created"

        result_df = pd.read_csv(submissions_path)
        assert len(result_df) == 6, f"Expected 6 rows but got {len(result_df)}"

        expected_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
        assert list(result_df.columns) == expected_cols


def test_concatenate_output_files_empty_dir():
    """
    Test concatenate_output_files creates an empty submissions.csv with correct columns
    when no files are found.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "empty_output")
        os.makedirs(out_dir, exist_ok=True)

        concatenate_output_files(out_dir)

        submissions_path = os.path.join(out_dir, 'submissions.csv')
        assert os.path.exists(submissions_path), "submissions.csv was not created"

        result_df = pd.read_csv(submissions_path)
        assert len(result_df) == 0
        expected_cols = ['ID', 'dataset', 'label_positive_probability', 'junction_aa', 'v_call', 'j_call']
        assert list(result_df.columns) == expected_cols


def test_load_and_encode_kmers_with_metadata():
    """
    Test load_and_encode_kmers with metadata.csv present.
    Verifies k-mer encoding with simple sequences for manual validation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir, exist_ok=True)

        rep1_df = pd.DataFrame({
            'junction_aa': ['AAA', 'AAB'],
            'v_call': ['TRBV1', 'TRBV2'],
            'j_call': ['TRBJ1', 'TRBJ2']
        })
        rep1_df.to_csv(os.path.join(data_dir, 'rep1.tsv'), sep='\t', index=False)

        rep2_df = pd.DataFrame({
            'junction_aa': ['ABCD', 'ABCE'],
            'v_call': ['TRBV3', 'TRBV4'],
            'j_call': ['TRBJ3', 'TRBJ4']
        })
        rep2_df.to_csv(os.path.join(data_dir, 'rep2.tsv'), sep='\t', index=False)

        metadata_df = pd.DataFrame({
            'repertoire_id': ['rep1', 'rep2'],
            'filename': ['rep1.tsv', 'rep2.tsv'],
            'label_positive': [True, False]
        })
        metadata_df.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)

        features_df, metadata_result = load_and_encode_kmers(data_dir, k=3)

        assert len(metadata_result) == 2
        assert set(metadata_result.columns) == {'ID', 'label_positive'}
        assert metadata_result['ID'].tolist() == ['rep1', 'rep2']
        assert metadata_result['label_positive'].tolist() == [True, False]

        assert len(features_df) == 2
        assert features_df.index.name == 'ID'
        assert 'rep1' in features_df.index
        assert 'rep2' in features_df.index

        rep1_row = features_df.loc['rep1']
        assert rep1_row.get('AAA', 0) == 1
        assert rep1_row.get('AAB', 0) == 1

        rep2_row = features_df.loc['rep2']
        assert rep2_row.get('ABC', 0) == 2
        assert rep2_row.get('BCD', 0) == 1
        assert rep2_row.get('BCE', 0) == 1


def test_load_and_encode_kmers_without_metadata():
    """
    Test load_and_encode_kmers without metadata.csv.
    Verifies that function works when only TSV files are present.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir, exist_ok=True)

        rep1_df = pd.DataFrame({
            'junction_aa': ['CCC', 'CCD'],
            'v_call': ['TRBV1', 'TRBV2'],
            'j_call': ['TRBJ1', 'TRBJ2']
        })
        rep1_df.to_csv(os.path.join(data_dir, 'sample1.tsv'), sep='\t', index=False)

        rep2_df = pd.DataFrame({
            'junction_aa': ['DDEE'],
            'v_call': ['TRBV3'],
            'j_call': ['TRBJ3']
        })
        rep2_df.to_csv(os.path.join(data_dir, 'sample2.tsv'), sep='\t', index=False)

        features_df, metadata_result = load_and_encode_kmers(data_dir, k=2)

        assert len(metadata_result) == 2
        assert 'ID' in metadata_result.columns
        assert 'label_positive' not in metadata_result.columns
        assert set(metadata_result['ID'].tolist()) == {'sample1', 'sample2'}

        assert features_df.index.name == 'ID'
        assert 'sample1' in features_df.index
        assert 'sample2' in features_df.index

        sample1_row = features_df.loc['sample1']
        assert sample1_row.get('CC', 0) == 3
        assert sample1_row.get('CD', 0) == 1

        sample2_row = features_df.loc['sample2']
        assert sample2_row.get('DD', 0) == 1
        assert sample2_row.get('DE', 0) == 1
        assert sample2_row.get('EE', 0) == 1


def test_load_and_encode_kmers_empty_sequences():
    """
    Test load_and_encode_kmers handles NaN/empty sequences gracefully.
    """
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir, exist_ok=True)

        rep_df = pd.DataFrame({
            'junction_aa': ['ABC', None, 'DEF', pd.NA],
            'v_call': ['TRBV1', 'TRBV2', 'TRBV3', 'TRBV4'],
            'j_call': ['TRBJ1', 'TRBJ2', 'TRBJ3', 'TRBJ4']
        })
        rep_df.to_csv(os.path.join(data_dir, 'rep1.tsv'), sep='\t', index=False)

        metadata_df = pd.DataFrame({
            'repertoire_id': ['rep1'],
            'filename': ['rep1.tsv'],
            'label_positive': [True]
        })
        metadata_df.to_csv(os.path.join(data_dir, 'metadata.csv'), index=False)

        features_df, metadata_result = load_and_encode_kmers(data_dir, k=2)

        assert features_df.index.name == 'ID'
        assert 'rep1' in features_df.index

        rep1_row = features_df.loc['rep1']
        assert rep1_row.get('AB', 0) == 1
        assert rep1_row.get('BC', 0) == 1
        assert rep1_row.get('DE', 0) == 1
        assert rep1_row.get('EF', 0) == 1


def test_get_dataset_pairs_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train")
        test_dir = os.path.join(tmpdir, "test")
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        # Create train datasets
        os.makedirs(os.path.join(train_dir, "train_dataset_1"))
        os.makedirs(os.path.join(train_dir, "train_dataset_2"))

        # Create test datasets (multiple for dataset 1)
        os.makedirs(os.path.join(test_dir, "test_dataset_1"))
        os.makedirs(os.path.join(test_dir, "test_dataset_1_1"))
        os.makedirs(os.path.join(test_dir, "test_dataset_2"))

        pairs = get_dataset_pairs(train_dir, test_dir)

        assert len(pairs) == 2
        assert pairs[0][0].endswith("train_dataset_1")
        assert len(pairs[0][1]) == 2
        assert pairs[1][0].endswith("train_dataset_2")
        assert len(pairs[1][1]) == 1


def test_get_dataset_pairs_multi_digit():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train")
        test_dir = os.path.join(tmpdir, "test")
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        os.makedirs(os.path.join(train_dir, "train_dataset_10"))
        os.makedirs(os.path.join(test_dir, "test_dataset_10"))
        os.makedirs(os.path.join(test_dir, "test_dataset_10_1"))

        pairs = get_dataset_pairs(train_dir, test_dir)

        assert len(pairs) == 1
        assert pairs[0][0].endswith("train_dataset_10")
        assert len(pairs[0][1]) == 2


def test_get_dataset_pairs_no_matching_tests():
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, "train")
        test_dir = os.path.join(tmpdir, "test")
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        os.makedirs(os.path.join(train_dir, "train_dataset_1"))

        pairs = get_dataset_pairs(train_dir, test_dir)

        assert len(pairs) == 1
        assert len(pairs[0][1]) == 0