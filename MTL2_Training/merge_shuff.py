import random

def shuffle_and_split_data(train_file, val_file, train_split_ratio=0.8):
    """
    Merges train and validation file lists, shuffles them, and splits them again
    based on the given train-validation ratio. It also ensures empty lines are not read or written.

    Args:
        train_file (str): Path to the training data file.
        val_file (str): Path to the validation data file.
        train_split_ratio (float): Proportion of data to use for training (default is 0.8).
    """

    # Read train and validation file contents, filtering out empty lines
    with open(train_file, "r") as f:
        train_lines = [line.strip() for line in f.readlines() if line.strip()]

    with open(val_file, "r") as f:
        val_lines = [line.strip() for line in f.readlines() if line.strip()]

    # Merge and shuffle
    all_data = train_lines + val_lines
    random.shuffle(all_data)

    # Split the data
    split_idx = int(len(all_data) * train_split_ratio)
    new_train_lines = all_data[:split_idx]
    new_val_lines = all_data[split_idx:]

    # Save back to files without trailing empty lines
    with open(train_file, "w") as f:
        f.writelines("\n".join(new_train_lines))

    with open(val_file, "w") as f:
        f.writelines("\n".join(new_val_lines))

    print(f"Dataset shuffled and split: {len(new_train_lines)} training samples, {len(new_val_lines)} validation samples.")    
