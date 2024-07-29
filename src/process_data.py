import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Загружаем данные в датасет -> скейл -> вывод последних 10 матчей команды

def process_data(data):
    proc = Processor()
    proc.scale_data(data)
    return data


def get_nba_dataloader(csv_path, batch_size=32, shuffle=True, num_workers=4, columns_A=None, columns_H=None):
    """
    Создает и возвращает DataLoader для NBADataset.

    Args:
    csv_path (str): Путь к CSV файлу с данными.
    batch_size (int): Размер батча. По умолчанию 32.
    shuffle (bool): Нужно ли перемешивать данные. По умолчанию True.
    num_workers (int): Количество процессов для загрузки данных. По умолчанию 4.
    columns_A (list): Список колонок для команды A. Если None, используются значения по умолчанию.
    columns_H (list): Список колонок для команды H. Если None, используются значения по умолчанию.

    Returns:
    torch.utils.data.DataLoader: DataLoader для NBADataset.
    """
    dataset = NBADataset(csv_path, columns_A, columns_H)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


class Processor():
    def scale_data(self, data): # Скейлит всю дату
        scaler = Scaler()
        scaler.fit(data)
        data = scaler.transform(data)
        return data


class Scaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= (self.std + 1e-7)
        return x

class NBADataset(Dataset):
    def __init__(self, root, columns_A=None, columns_H=None):
        if columns_A is None:
            columns_A = ['a_eFGp', 'a_FTr', 'a_ORBp', 'a_TOVp']
        if columns_H is None:
            columns_H = ['h_eFGp', 'h_FTr', 'h_ORBp', 'h_TOVp']
        self.root = root
        self.df = pd.read_csv(root)
        self.columns_A = columns_A
        self.columns_H = columns_H

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        away_score = self.df.loc[idx, "away_score"]
        home_score = self.df.loc[idx, "home_score"]
        result = self.df.loc[idx, "result"]
        team1_stats = torch.tensor(self.df.loc[idx, self.columns_A])
        team2_stats = torch.tensor(self.df.loc[idx, self.columns_H])

        train_data = torch.cat((team1_stats, team2_stats), dim=0)
        train_data = process_data(train_data)

        sample = {'stats': train_data, 'results': [result, away_score, home_score]}

        return sample

