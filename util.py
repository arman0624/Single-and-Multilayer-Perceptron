from typing import List, Any, Tuple

from newsgroups import newsgroups_data_loader
from sst2 import sst2_data_loader
from collections import Counter


def save_results(predictions: List[Any], results_path: str) -> None:
    """ Saves the predictions to a file.

    Inputs:
        predictions (list of predictions, e.g., string)
        results_path (str): Filename to save predictions to
    """
    num_to_label = {0: 'alt.atheism',
                    1: 'comp.graphics',
                    2: 'comp.sys.mac.hardware',
                    3: 'comp.sys.ibm.pc.hardware',
                    4: 'comp.os.ms-windows.misc',
                    5: 'comp.windows.x',
                    6: 'misc.forsale',
                    7: 'rec.autos',
                    8: 'rec.motorcycles',
                    9: 'rec.sport.baseball',
                    10: 'rec.sport.hockey',
                    11: 'sci.med',
                    12: 'sci.crypt',
                    13: 'sci.space',
                    14: 'sci.electronics',
                    15: 'soc.religion.christian',
                    16: 'talk.religion.misc',
                    17: 'talk.politics.mideast',
                    18: 'talk.politics.misc',
                    19: 'talk.politics.guns'
                    }

    with open(results_path, 'w') as file:
        id = 0
        if "newsgroups" in results_path:
            file.write("id,newsgroup\n")
        else:
            file.write("id,label\n")
        for p in predictions:
            if "mlp" in results_path:
                if "newsgroups" in results_path:
                    file.write(str(id) + "," + num_to_label[p.item()] + "\n")
                else:
                    file.write(str(id) + "," + str(p.item()) + "\n")
            else:
                if "newsgroups" in results_path:
                    file.write(str(id) + "," + num_to_label[p] + "\n")
                else:
                    file.write(str(id) + "," + str(p) + "\n")
            id += 1


def compute_accuracy(labels: List[Any], predictions: List[Any]) -> float:
    """ Computes the accuracy given some predictions and labels.

    Inputs:
        labels (list): Labels for the examples.
        predictions (list): The predictions.
    Returns:
        float representing the % of predictions that were true.
    """
    if len(labels) != len(predictions):
        raise ValueError("Length of labels (" + str(len(labels)) + " not the same as length of predictions (" + str(len(predictions)) + ".")
    correct_preds = 0
    for i in range(len(predictions)):
        if int(labels[i]) == int(predictions[i]):
            correct_preds += 1
    return float(correct_preds/len(predictions))


def evaluate(model: Any, data: List[Tuple[Any, Any]], results_path: str) -> float:
    """ Evaluates a dataset given the model.

    Inputs:
        model: A model with a prediction function.
        data: Suggested type is (list of pair), where each item is a training
            examples represented as an (input, label) pair. And when using the
            test data, your label can be some null value.
        results_path (str): A filename where you will save the predictions.
    """

    predictions = [model.predict(example[0]) for example in data]
    save_results(predictions, results_path)

    return compute_accuracy([int(example[1]) for example in data], predictions)


def load_data(data_type: str, feature_type: str, model_type: str):
    """ Loads the data.

    Inputs:
        data_type: The type of data to load.
        feature_type: The type of features to use.
        model_type: The type of model to use.

    Returns:
        Training, validation, development, and testing data, as well as which kind of data
            was used.
    """
    data_loader = None
    if data_type == "newsgroups":
        data_loader = newsgroups_data_loader
    elif data_type == "sst2":
        data_loader = sst2_data_loader

    assert data_loader, "Choose between newsgroups or sst2 data. " \
                        + "data_type was: " + str(data_type)

    # Load the data.
    train_data, val_data, dev_data, test_data = data_loader("data/" + data_type + "/train/train_data.csv",
                                                            "data/" + data_type + "/train/train_labels.csv",
                                                            "data/" + data_type + "/dev/dev_data.csv",
                                                            "data/" + data_type + "/dev/dev_labels.csv",
                                                            "data/" + data_type + "/test/test_data.csv",
                                                            feature_type,
                                                            model_type)

    return train_data, val_data, dev_data, test_data
