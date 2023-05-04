import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size, shuffle=True)
    
    modelQualityTracker = {"Training Losses Per Epoch":[], "ValidationInformation":[], "Accuracies":[]}
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    #Extraneous lines of code I don't want to get rid of yet: train_losses = []; step = 0;
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        for batchInputs, batchLabels in tqdm(train_loader): #what does tdqm stand for?
            optimizer.zero_grad()
            modelPredictionsForLabels = model(batchInputs).squeeze()
            lossForCurrentBatch = loss_fn(modelPredictionsForLabels, batchLabels)
            lossForCurrentBatch.backward()
            optimizer.step()
            modelQualityTracker["Training Losses Per Epoch"].append(lossForCurrentBatch.data.item())
        #Evaluate our model and log to Tensorboard (see past version for template code for doing this)
    
    modelQualityTracker["Average Loss for Each Epoch"] = ([sum(i)/len(i) for i in modelQualityTracker["Training Losses Per Epoch"]])
    modelQualityTracker["Accuracies"].append(compute_accuracy(modelPredictionsForLabels, batchLabels))
    return modelQualityTracker

def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    total_loss = 0.0
    for inputs, targets in tqdm(val_loader):
        outputs = model(inputs)
        loss = loss_fn(inputs, targets)
        total_loss += loss/outputs.len

    pass
