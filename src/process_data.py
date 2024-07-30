import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class Scaler:
    """
    Scaler is a class for scaling data using standard scaling (mean=0, std=1).

    Attributes:
    mean (numpy.ndarray): The mean of the data.
    std (numpy.ndarray): The standard deviation of the data.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        """
        Calculates the mean and standard deviation of the data.

        Parameters:
        x (numpy.ndarray): The data to calculate the mean and standard deviation from.
        """
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        """
        Scales the data using standard scaling (mean=0, std=1).

        Parameters:
        x (numpy.ndarray): The data to scale.

        Returns:
        numpy.ndarray: The scaled data.
        """
        return (x - self.mean) / (self.std + 1e-7)


class Processor:
    """
    Processor is a class for processing data using a Scaler.

    Attributes:
    scaler (Scaler): The Scaler to use for processing the data.
    """
    def __init__(self):
        self.scaler = Scaler()

    def scale_data(self, data):
        """
        Scales the data using the Scaler.

        Parameters:
        data (numpy.ndarray): The data to scale.

        Returns:
        numpy.ndarray: The scaled data.
        """
        self.scaler.fit(data)
        return self.scaler.transform(data)


def process_data(data):
    """
    Processes the data using a Processor.

    Parameters:
    data (numpy.ndarray): The data to process.

    Returns:
    numpy.ndarray: The processed data.
    """
    processor = Processor()
    return processor.scale_data(data)


class NBADataset(Dataset):
    """
    NBADataset is a class for creating a dataset from NBA data.

    Attributes:
    columns_A (list): The columns for the away team.
    columns_H (list): The columns for the home team.
    df (pandas.DataFrame): The DataFrame containing the NBA data.
    augment (bool): Whether to augment the data or not.
    """
    def __init__(self, root, columns_A=None, columns_H=None, augment=False):
        """
        Initializes the NBADataset class with the given root, columns for the away and home teams, and whether to augment the data or not.

        Parameters:
        root (str): The path to the NBA data.
        columns_A (list): The columns for the away team.
        columns_H (list): The columns for the home team.
        augment (bool): Whether to augment the data or not.
        """
        self.columns_A = columns_A or ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp']
        self.columns_H = columns_H or ['h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']
        self.df = pd.read_csv(root)

        all_columns = self.columns_A + self.columns_H + ["away_score", "home_score", "result"]
        self.df[all_columns] = self.df[all_columns].fillna(self.df[all_columns].mean())

        self.augment = augment

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
        int: The length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Gets the item at the given index.

        Parameters:
        idx (int): The index to get the item from.

        Returns:
        dict: A dictionary containing the stats and results.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        away_score = float(self.df.loc[idx, "away_score"])
        home_score = float(self.df.loc[idx, "home_score"])

        team1_stats = self.df.loc[idx, self.columns_A].astype(float).values
        team2_stats = self.df.loc[idx, self.columns_H].astype(float).values

        if self.augment:
            team1_stats = self.augment_data(team1_stats)
            team2_stats = self.augment_data(team2_stats)

        team1_stats = torch.tensor(team1_stats, dtype=torch.float32)
        team2_stats = torch.tensor(team2_stats, dtype=torch.float32)

        train_data = torch.cat((team1_stats, team2_stats), dim=0)
        train_data = process_data(train_data)

        return {
            'stats': train_data,
            'results': torch.tensor([away_score, home_score], dtype=torch.float32)
        }

    def augment_data(self, stats):
        """
        Augments the data by adding small random noise.

        Parameters:
        stats (numpy.ndarray): The data to augment.

        Returns:
        numpy.ndarray: The augmented data.
        """
        # Add small random noise to the stats
        noise = np.random.normal(0, 0.01, stats.shape)
        return stats + noise


def get_nba_dataloader(csv_path, batch_size=32, shuffle=True, num_workers=4, columns_A=None, columns_H=None):
    """
    Gets a DataLoader for the NBA data.

    Parameters:
    csv_path (str): The path to the NBA data.
    batch_size (int): The batch size for the DataLoader.
    shuffle (bool): Whether to shuffle the data or not.
    num_workers (int): The number of workers for the DataLoader.
    columns_A (list): The columns for the away team.
    columns_H (list): The columns for the home team.

    Returns:
    torch.utils.data.DataLoader: The DataLoader for the NBA data.
    """
    dataset = NBADataset(csv_path, columns_A, columns_H)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
