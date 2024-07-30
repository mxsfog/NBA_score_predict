import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from src.Model import NBAModel
from src.process_data import NBADataset


class CustomLoss(nn.Module):
    """
    CustomLoss is a class for calculating the custom loss function which is a combination of Mean Squared Error (MSE)
    and Mean Absolute Error (MAE).
    """
    def __init__(self):
        """
        Initializes the CustomLoss class with MSE and MAE loss functions.
        """
        super(CustomLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, outputs, targets):
        """
        Calculates the custom loss, MSE and MAE for the given outputs and targets.

        Parameters:
        outputs (torch.Tensor): The outputs from the model.
        targets (torch.Tensor): The actual targets.

        Returns:
        tuple: A tuple containing the custom loss, MSE and MAE.
        """
        mse_loss = self.mse(outputs[:, :2], targets[:, :2])
        mae_loss = self.mae(outputs[:, :2], targets[:, :2])
        return mse_loss + mae_loss, mse_loss, mae_loss


def calculate_metrics(y_true, y_pred):
    """
    Calculates the MSE, MAE and R2 score for the given true and predicted values.

    Parameters:
    y_true (numpy.ndarray): The actual values.
    y_pred (numpy.ndarray): The predicted values.

    Returns:
    tuple: A tuple containing the MSE, MAE and R2 score.
    """
    mse = mean_squared_error(y_true[:, :2], y_pred[:, :2])
    mae = mean_absolute_error(y_true[:, :2], y_pred[:, :2])
    r2 = r2_score(y_true[:, :2], y_pred[:, :2])
    return mse, mae, r2


def validate_model(model, dataloader, criterion, device):
    """
    Validates the model on the given dataloader and calculates the average loss, MSE and MAE.

    Parameters:
    model (torch.nn.Module): The model to validate.
    dataloader (torch.utils.data.DataLoader): The dataloader to validate on.
    criterion (torch.nn.Module): The loss function.
    device (torch.device): The device to run the validation on.

    Returns:
    tuple: A tuple containing the average loss, MSE, MAE and the calculated metrics.
    """
    model.eval()
    total_loss = 0
    total_mse_loss = 0
    total_mae_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            stats = batch['stats'].to(device)
            results = batch['results'].to(device)

            outputs = model(stats)
            loss, mse_loss, mae_loss = criterion(outputs, results)

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_mae_loss += mae_loss.item()
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(results.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_mse_loss = total_mse_loss / len(dataloader)
    avg_mae_loss = total_mae_loss / len(dataloader)
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    mse, mae, r2 = calculate_metrics(all_targets, all_predictions)

    return avg_loss, avg_mse_loss, avg_mae_loss, mse, mae, r2


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    """
    Trains the model for a specified number of epochs.

    Parameters:
    model (torch.nn.Module): The model to train.
    train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
    val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
    num_epochs (int): The number of epochs to train the model.
    learning_rate (float): The learning rate for the optimizer.
    device (torch.device): The device to run the training on.

    Returns:
    None
    """
    criterion = CustomLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    model.to(device)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            stats = batch['stats'].to(device)
            results = batch['results'].to(device)

            optimizer.zero_grad()
            outputs = model(stats)
            _, _, loss = criterion(outputs, results)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        val_loss, val_mse_loss, val_mae_loss, val_mse, val_mae, val_r2 = validate_model(model, val_loader,
                                                                                        criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, MSE Loss: {val_mse_loss:.4f}, MAE Loss: {val_mae_loss:.4f}')
        print(f'Val MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, R2: {val_r2:.4f}')

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "D:/NBAAnalyse/models/best_nba_transformer_model.pth")
            print("Saved best model")

        print("-" * 50)

    print("Training completed!")


# Example of usage:
if __name__ == "__main__":
    """
    Main function to run the training process.
    """
    csv_path = "D:/NBAAnalyse/Data/team_factor_10.csv"

    # Splitting the data into training and validation sets
    full_dataset = NBADataset(csv_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Model parameters
    input_dim = 8  # Total number of features for both teams
    nhead = 2
    num_encoder_layers = 2
    dim_feedforward = 64
    dropout = 0.1

    # Creating the model
    model = NBAModel(input_dim, num_encoder_layers, dim_feedforward, dropout)

    # Training parameters
    num_epochs = 1000
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training the model
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # Loading the best model
    model.load_state_dict(torch.load("D:/NBAAnalyse/models/best_nba_transformer_model.pth"))

    # Final validation
    criterion = CustomLoss()
    final_loss, final_mse_loss, final_mae_loss, final_mse, final_mae, final_r2 = validate_model(model, val_loader,
                                                                                                criterion, device)
    print("Final Validation Results:")
    print(f'Loss: {final_loss:.4f}, MSE Loss: {final_mse_loss:.4f}, MAE Loss: {final_mae_loss:.4f}')
    print(f'MSE: {final_mse:.4f}, MAE: {final_mae:.4f}, R2: {final_r2:.4f}')