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

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        losses = []

        # Loop over each batch in the dataset
        for inputs, targets in tqdm(train_loader):
            model.zero_grad()
            # TODO: Forward propagate
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets) #targets.float?
            # TODO: Backpropagation and gradient descent
            loss.backward()
            optimizer.step()
            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                train_loss_value = sum(train_losses)/len(train_losses)# Compute training loss and accuracy.
                # Log the results to Tensorboard.
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn)

            losses.append(loss.item())
            step += 1

        epoch_loss = sum(losses)/step
        train_losses.append(loss.item())
        print()


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
