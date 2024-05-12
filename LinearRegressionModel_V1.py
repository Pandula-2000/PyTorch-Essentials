import pandas as pd
import numpy
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.modules.module import T

# print(torch.__version__)

# ------------------------------------------- Let's Create Data. -------------------------------------------------------
weight = 0.7
bias = 0.3

start, end, step = 0, 1, 0.02

X = torch.arange(start, end, step).unsqueeze(dim=1)
Y = weight * X + bias

# ------------------------------------------- Split train, test Data.---------------------------------------------------
train_split = int(0.8 * len(X))
X_train, Y_train = X[:train_split], Y[:train_split]
X_test, Y_test = X[train_split:], Y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=Y_train,
                     test_data=X_test,
                     test_labels=Y_test,
                     predictions=None):
    plt.figure(figsize=(10, 10))
    plt.scatter(train_data, train_labels, c='b', s=5, label='Train Data')
    plt.scatter(test_data, test_labels, c='g', s=5, label='Test Data')
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=5, label='Predictions')
    plt.legend(prop={'size': 14})
    plt.show()


plot_predictions()


# -------------------------------------------- Build model. ------------------------------------------------------------

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1,
                                               dtype=torch.float,
                                               requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1,
                                             dtype=torch.float))

    # To define the forward computation of the model,
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.weight * X + self.bias


# Set the random Seed.
torch.manual_seed(42)

# ------------------------------------------ Create Model instance.-----------------------------------------------------
model = LinearRegressionModel()

# To check the model parameters (weight, bias) you can use,
print(list(model.parameters()))
# We can also get the state dict as,
print(model.state_dict())

# ------------------------------------------- Make Predictions.---------------------------------------------------------

with torch.inference_mode():
    # Runs data through forward()
    y_preds = model(X_test)

plot_predictions(predictions=y_preds)

# --------------------------------- Train and Test Loop of the Model.---------------------------------------------------

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

epochs = 200
epoch_count = []
train_loss_history = []
test_loss_history = []

for epoch in range(epochs):
    # ----------------------------------- TRAIN LOOP -------------------------------------
    # Put the model in training mode
    model.train()
    # Forward pass
    y_pred = model(X_train)
    # Calculate the loss.
    loss = loss_fn(y_pred, Y_train)
    # Reset (0) the gradients.
    optimizer.zero_grad()
    # BackPropagation.
    loss.backward()
    # Gradient descent.
    optimizer.step()

    # ---------------------------------- TEST LOOP ----------------------------------------
    # Put the model in eval mode.
    model.eval()
    # Turn on the inference mode.
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred, Y_test)

    # Let's print what is happening at every 10 epochs.

    if epoch % 10 == 0:
        epoch_count.append(epoch)
        train_loss_history.append(loss.detach().numpy())
        test_loss_history.append(test_loss.detach().numpy())
        print(f'Epoch: {epoch}, L1 Train Loss: {loss:.4f}, L1 Test Loss: {test_loss:.4f}')

# --------------------------------------------- PLOT LOSSES ------------------------------------------------------------

plt.plot(epoch_count, train_loss_history, label='Train Loss')
plt.plot(epoch_count, test_loss_history, label='Test Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

print("")
print(model.state_dict())

# -------------------------------- MAKING PREDICTIONS WITH TRAINED MODEL -----------------------------------------------

# Set the model to the evaluation mode.
model.eval()

with torch.inference_mode():
    """
    Make sure that the calculations are done with model and data on the same device (CPU/GPU).
    >> model.to(device)
    >> y_pred.to(device)
    >> X_pred.to(device)
    """
    y_final_pred = model(X_test)

plot_predictions(predictions=y_final_pred)

# ---------------------------------------------- PICKLE THE MODEL ------------------------------------------------------

from pathlib import Path

# Create Model directory.
MODEL_PATH = Path('Models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Create model save path.
MODEL_NAME = 'LinearRegressionModel_v1.pth'
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model.
torch.save(model.state_dict(), f=MODEL_SAVE_PATH)

# To Load a saved model, create an instance of LinearRegressionModel.
loaded_model = LinearRegressionModel()
# Then load the state_dist with pickled values.
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

