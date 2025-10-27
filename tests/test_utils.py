import os
import tempfile
import pytest
from submission.utils import validate_dirs_and_files


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


def test_out_dir_exists():
    with tempfile.TemporaryDirectory() as tmp:
        train_dir = os.path.join(tmp, "train")
        test_dir = os.path.join(tmp, "test")
        out_dir = os.path.join(tmp, "out")
        create_dir_with_tsv_and_metadata(train_dir)
        create_dir_with_tsv(test_dir)
        os.makedirs(out_dir, exist_ok=True)
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
