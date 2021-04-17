import pickle

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import KFold


class RelationDataset(Dataset):
    label_dict_file = "/opt/ml/input/data/label_type.pkl"
    train_file = "/opt/ml/input/data/train/train.tsv"
    test_file = "/opt/ml/input/data/test/test.tsv"

    def __init__(self, tokenizer, max_length=512, train=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train = train
        self.data = self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.tokenizer(self.data.loc[index, "col1"], padding="max_length", max_length=self.max_length)
        entity_indices = self.get_entity_indices(data["input_ids"])
        if self.train:
            label = torch.tensor(self.data.loc[index, "label"], dtype=torch.long)
        else:
            label = torch.tensor(0)
        return (torch.tensor(data["input_ids"]), torch.tensor(data["token_type_ids"]),
                torch.tensor(data["attention_mask"]), torch.tensor(entity_indices), label)

    def get_data(self):
        path = self.train_file if self.train else self.test_file
        df = pd.read_csv(path, sep='\t', names=["col%i" % i for i in range(9)])
        if self.train:
            label_dict = self.get_label_dict()
            df["label"] = df["col8"].map(label_dict)
        self.insert_entity_token(df)
        return df

    def insert_entity_token(self, df):
        for idx, row in df.iterrows():
            sent = df.loc[idx, "col1"]
            sent = list(sent)
            if row["col3"] < row["col6"]:
                sent.insert(df.loc[idx, "col7"] + 1, " [/ENT] ")
                sent.insert(df.loc[idx, "col6"], " [ENT] ")
                sent.insert(df.loc[idx, "col4"] + 1, " [/ENT] ")
                sent.insert(df.loc[idx, "col3"], " [ENT] ")
            else:
                sent.insert(df.loc[idx, "col4"] + 1, " [/ENT] ")
                sent.insert(df.loc[idx, "col3"], " [ENT] ")
                sent.insert(df.loc[idx, "col7"] + 1, " [/ENT] ")
                sent.insert(df.loc[idx, "col6"], " [ENT] ")
            df.loc[idx, "col1"] = "".join(sent)
        return

    def get_label_dict(self):
        with open(self.label_dict_file, "rb") as f:
            return pickle.load(f)

    def get_entity_indices(self, input_ids):
        entity_indices = []
        entity_id = self.tokenizer.convert_tokens_to_ids("[ENT]")
        padding_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        for idx, input_id in enumerate(input_ids):
            if input_id == entity_id:
                entity_indices.append(idx)
            elif input_ids == padding_id:
                break
        return entity_indices


def k_fold_dataset(dataset, k=5, random_state=1818):
    k_fold = KFold(k, shuffle=True, random_state=random_state)
    dummy_x = [0] * len(dataset)
    idx = 0
    for train_idx, val_idx in k_fold.split(dummy_x):
        yield idx, Subset(dataset, train_idx), Subset(dataset, val_idx)
        idx += 1
