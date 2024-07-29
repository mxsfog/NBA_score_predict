import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        return (x - self.mean) / (self.std + 1e-7)

class Processor:
    def __init__(self):
        self.scaler = Scaler()

    def scale_data(self, data):
        self.scaler.fit(data)
        return self.scaler.transform(data)

def process_data(data):
    processor = Processor()
    return processor.scale_data(data)

class NBADataset(Dataset):
    def __init__(self, root, columns_A=None, columns_H=None):
        self.columns_A = columns_A or ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp']
        self.columns_H = columns_H or ['h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']
        self.df = pd.read_csv(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        away_score = self.df.loc[idx, "away_score"]
        home_score = self.df.loc[idx, "home_score"]
        result = self.df.loc[idx, "result"]
        team1_stats = torch.tensor(self.df.loc[idx, self.columns_A].values, dtype=torch.float32)
        team2_stats = torch.tensor(self.df.loc[idx, self.columns_H].values, dtype=torch.float32)

        train_data = torch.cat((team1_stats, team2_stats), dim=0)
        train_data = process_data(train_data)

        return {
            'stats': train_data,
            'results': torch.tensor([result, away_score, home_score], dtype=torch.float32)
        }

def get_nba_dataloader(csv_path, batch_size=32, shuffle=True, num_workers=4, columns_A=None, columns_H=None):
    dataset = NBADataset(csv_path, columns_A, columns_H)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
