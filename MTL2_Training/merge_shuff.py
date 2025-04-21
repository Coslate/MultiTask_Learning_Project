import random
import os

def shuffle_and_split_data(train_file, val_file, test_file, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05):
    """
    Merges train and validation file lists, shuffles them, and splits into
    train, val, and test sets with specified ratios. Supports empty-line filtering.

    Args:
        train_file (str): Path to save the training split.
        val_file (str): Path to save the validation split.
        test_file (str): Path to save the test split.
        train_ratio (float): Proportion of data for training (default: 0.9).
        val_ratio (float): Proportion of data for validation (default: 0.05).
        test_ratio (float): Proportion of data for testing (default: 0.05).
    """

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Read and merge all lines from original train/val files
    with open(train_file, "r") as f:
        train_lines = [line.strip() for line in f if line.strip()]
    with open(val_file, "r") as f:
        val_lines = [line.strip() for line in f if line.strip()]

    if test_file is not None and os.path.exists(test_file):
        with open(test_file, "r") as f:
            test_lines = [line.strip() for line in f if line.strip()]
        all_data = train_lines + val_lines + test_lines
    else:
        all_data = train_lines + val_lines
    
    random.shuffle(all_data)
    total = len(all_data)

    # Split the data
    train_idx = int(len(all_data) * train_ratio)
    val_idx = int(len(all_data)*val_ratio)
    test_idx = total - train_idx - val_idx
    new_train_lines = all_data[:train_idx]
    new_val_lines = all_data[train_idx:(train_idx+val_idx)]
    new_test_lines = all_data[(train_idx+val_idx):]

    # Save back to files without trailing empty lines
    with open(train_file, "w") as f:
        f.writelines("\n".join(new_train_lines))

    with open(val_file, "w") as f:
        f.writelines("\n".join(new_val_lines))

    with open(test_file, "w") as f:
        f.writelines("\n".join(new_test_lines))

    print(f"Split complete: {len(new_train_lines)} train | {len(new_val_lines)} val | {len(new_test_lines)} test samples")
