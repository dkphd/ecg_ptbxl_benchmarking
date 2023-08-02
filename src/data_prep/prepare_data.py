from src.utils import utils

from argparse import ArgumentParser


def prepare(
    datafolder,
    outputfolder,
    task="diagnostic",
    sampling_frequency=100,
    min_samples=0,
    train_fold=8,
    val_fold=9,
    test_fold=10,
    folds_type="strat",
):
    # Load PTB-XL data
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)

    # Preprocess label data
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)

    # Select relevant data and convert to one-hot
    data, labels, Y, _ = utils.select_data(
        data, labels, task, min_samples, outputfolder + "/data"
    )
    input_shape = data[0].shape

    # 10th fold for testing (9th for now)
    X_test = data[labels.strat_fold == test_fold]
    y_test = Y[labels.strat_fold == test_fold]
    # 9th fold for validation (8th for now)
    X_val = data[labels.strat_fold == val_fold]
    y_val = Y[labels.strat_fold == val_fold]
    # rest for training
    X_train = data[labels.strat_fold <= train_fold]
    y_train = Y[labels.strat_fold <= train_fold]

    # Preprocess signal data
    X_train, X_val, X_test = utils.preprocess_signals(
        X_train, X_val, X_test, outputfolder
    )
    n_classes = y_train.shape[1]

    # save train and test labels
    y_train.dump(outputfolder + "/y_train.npy")
    y_val.dump(outputfolder + "/y_val.npy")
    y_test.dump(outputfolder + "/y_test.npy")

    # save train and test features
    X_train.dump(outputfolder + "/X_train.npy")
    X_val.dump(outputfolder + "/X_val.npy")
    X_test.dump(outputfolder + "/X_test.npy")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_dir")
    parser.add_argument("-o", "--output_dir")

    args = parser.parse_args()

    prepare(args.input_dir, args.output_dir)
