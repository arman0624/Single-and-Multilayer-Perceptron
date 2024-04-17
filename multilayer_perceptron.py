""" Multi-layer perceptron model
"""
import os
import argparse
from torch.utils.data import DataLoader, TensorDataset
from util import evaluate, load_data
import torch
import torch.nn as nn
import torch.optim


class MultilayerPerceptronModel(nn.Module):
    """ Multi-layer perceptron model for classification.
    """
    def __init__(self, num_classes, vocab_size):
        """ Initializes the model.

        Inputs:
            num_classes (int): The number of classes.
            vocab_size (int): The size of the vocabulary.
        """
        super(MultilayerPerceptronModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size+1, 128)
        self.hidden = nn.Linear(128, 256)
        self.activation = nn.ReLU()
        self.activation2 = nn.Tanh()  
        self.hidden2 = nn.Linear(256, 64)
        self.activation3 = nn.Tanh()
        self.activation4 = nn.ReLU()  
        self.output = nn.Linear(64, num_classes)

    def forward(self, model_input):
        y = model_input.long()
        y = self.embedding(y)
        y = torch.mean(y, dim=1)
        y = self.hidden(y)
        y = self.activation(y)
        y = torch.tanh(y)
        y = self.hidden2(y)
        y = torch.tanh(y)
        y = self.activation4(y)
        y = self.output(y)
        return y

    def predict(self, model_input: torch.Tensor):
        """ Predicts a label for an input.

        Inputs:
            model_input (tensor): Input data for an example or a batch of examples.

        Returns:
            The predicted class.

        """
        self.eval()
        with torch.no_grad():
            _, predicted_classes = torch.max(self.forward(model_input.long().unsqueeze(0)), dim=1)
        return predicted_classes

    def data_load(self, data, bsize):
        dtensor = torch.stack([d[0] for d in data])
        ltensor = torch.tensor([(d[1]) for d in data])
        data_set = TensorDataset(dtensor, ltensor)
        d_loader = DataLoader(data_set, batch_size=bsize, shuffle=True)
        return d_loader

    def learn(self, training_data, val_data, loss_fct, optimizer, num_epochs, lr) -> None:
        """ Trains the MLP.

        Inputs:
            training_data: Suggested type for an individual training example is 
                an (input, label) pair or (input, id, label) tuple.
                You can also use a dataloader.
            val_data: Validation data.
            loss_fct: The loss function.
            optimizer: The optimization method.
            num_epochs: The number of training epochs.
        """
        x_train = self.data_load(training_data, bsize=1024)
        x_val = self.data_load(val_data, bsize=1024)
        for i in range(num_epochs):
            train_loss, correct, total = 0.0, 0, 0
            for data, label in x_train:
                optimizer.zero_grad()
                y = self.forward(data)
                loss = loss_fct(y, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                _, predicted = torch.max(y.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            train_accuracy = 100 * correct / total

            with torch.no_grad():
                val_loss, correct, total = 0.0, 0, 0
                for data, label in x_val:
                    y = self.forward(data)
                    loss = loss_fct(y, label)
                    val_loss += loss.item()
                    _, predicted = torch.max(y, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                val_acc = 100 * correct / total

            print(f"Epoch {i}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MultiLayerPerceptron model')
    parser.add_argument('-d', '--data', type=str, default='sst2',
                        help='Dataset')
    parser.add_argument('-f', '--features', type=str, default='feature_name', help='Feature type')
    parser.add_argument('-m', '--model', type=str, default='mlp', help='Model type')
    args = parser.parse_args()

    data_type = args.data
    feature_type = args.features
    model_type = args.model

    train_data, val_data, dev_data, test_data = load_data(data_type, feature_type, model_type)

    full_train_data = [data for data, _ in train_data] + [data for data, _ in val_data]
    v_size = torch.cat(full_train_data).max().item()
    vocab_size = int(v_size) + 1

    # Train the model using the training data.
    if data_type == "sst2":
        num_classes = 2
    else:
        num_classes = 20

    model = MultilayerPerceptronModel(num_classes, vocab_size)
    loss_fct = nn.CrossEntropyLoss()
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 50
    print("Training the model...")
    model.learn(train_data, val_data, loss_fct, optimizer, num_epochs, lr)

    # Predict on the development set.
    dev_accuracy = evaluate(model,
                            dev_data,
                            os.path.join("results", f"mlp_{data_type}_{feature_type}_dev_predictions.csv"))

    print(dev_accuracy)
    # Predict on the test set.
    evaluate(model,
             test_data,
             os.path.join("results", f"mlp_{data_type}_{feature_type}_test_predictions.csv"))
