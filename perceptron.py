""" Perceptron model
"""
import os
import sys
import argparse
from typing import Dict, List

from util import evaluate, load_data, compute_accuracy


class PerceptronModel():
    """ Perceptron model for classification.
    """
    def __init__(self, num_features: int, num_classes: int):
        """ Initializes the model.

        Inputs:
            num_features (int): The number of features.
            num_classes (int): The number of classes.
        """
        self.weights: Dict[int, Dict[int, float]] = {}
        self.num_features = num_features
        self.num_classes = num_classes
        for class_id in range(num_classes):
            self.weights[class_id] = {}
            for feature in range(num_features):
                self.weights[class_id][feature] = 0.0

    def score(self, model_input: Dict, class_id: int):
        """ Compute the score of a class given the input.

        Inputs:
            model_input (features): Input data for an example
            class_id (int): Class id.

        Returns:
            The output score.
        """
        score = 0.0
        for feature, value in model_input.items():
            score += self.weights[class_id][feature] * value
        return score

    def predict(self, model_input: Dict) -> int:
        """ Predicts a label for an input.

        Inputs:
            model_input (features): Input data for an example

        Returns:
            The predicted class.
        """
        pred_c = None
        max_val = float('-inf')
        for class_id in self.weights.keys():
            pred_val = self.score(model_input, class_id)
            if pred_val > max_val:
                max_val = pred_val
                pred_c = class_id
        return pred_c

    def update_parameters(self, model_input: Dict, prediction: int, target: int, lr: float) -> None:
        """ Update the model weights of the model using the perceptron update rule.

        Inputs:
            model_input (features): Input data for an example
            prediction: The predicted label.
            target: The true label.
            lr: Learning rate.
        """
        if target != prediction:
            for feature, value in model_input.items():
                self.weights[int(target)][feature] += lr * value
                self.weights[prediction][feature] -= lr * value

    def learn(self, training_data, val_data, num_epochs, lr) -> None:
        """ Perceptron model training.

        Inputs:
            training_data: Suggested type is (list of tuple), where each item can be
                a training example represented as an (input, label) pair or (input, id, label) tuple.
            val_data: Validation data.
            num_epochs: Number of training epochs.
            lr: Learning rate.
        """
        best_weights = self.weights
        best_accuracy = 0.0
        for i in range(num_epochs):
            print("Epoch: ", i)
            predictions = []
            labels = []
            for data, label in training_data:
                pred = self.predict(data)
                self.update_parameters(data, pred, label, lr)
                predictions.append(pred)
                labels.append(label)
            acc = compute_accuracy(labels, predictions)
            if acc < best_accuracy:
                self.weights = best_weights
                break
            best_accuracy = acc
            best_weights = self.weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='perceptron', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model

    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)
    # Train the model using the training data.
    if data_type == "sst2":
        num_classes = 2
    else:
        num_classes = 20
    num_features = len(train_data[0][0])
    lr = 0.8
    num_epochs = 5

    model = PerceptronModel(num_features, num_classes)

    print("Training the model...")
    model.learn(train_data, val_data, num_epochs, lr)

    # Predict on the development set.
    print("Predict on the development set.")
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", f"perceptron_{data_type}_{feature_type}_dev_predictions.csv"))

    print(dev_accuracy)

    print("Predict on the test set.")
    # Predict on the test set.
    evaluate(model,
             test_data,
             os.path.join("results", f"perceptron_{data_type}_{feature_type}_test_predictions.csv"))
