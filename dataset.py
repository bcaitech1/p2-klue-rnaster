import pickle

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, Subset


def get_entity_dataset(version, file_path, tokenizer, **kwargs):
    if version == "v1":
        return EntityDataset(file_path, tokenizer, **kwargs)
    if version == "v2":
        return EntityDatasetV2(file_path, tokenizer, **kwargs)
    if version == "v3":
        return EntityDatasetV3(file_path, tokenizer, **kwargs)
    raise ValueError("incorrect version: %s" % version)


class EntityDataset(Dataset):
    label_dict_file = "/opt/ml/input/data/label_type.pkl"

    def __init__(self, file_path, tokenizer, max_length=512, is_train=True):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.data = self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sent = self.get_sent_with_entity_token(index)
        inputs = self.tokenizer(sent,
                                padding="max_length",
                                max_length=self.max_length,
                                return_tensors="pt")
        inputs = {key: val.squeeze() for key, val in inputs.items()}
        entity_token_indices = self.get_entity_token_indices(inputs["input_ids"])
        if self.is_train:
            label = torch.tensor(self.data.loc[index, "label"], dtype=torch.long)
        else:
            label = torch.tensor(0)
        inputs.update({"entity_token_indices": entity_token_indices, "labels": label})
        return inputs

    def get_data(self):
        df = pd.read_csv(self.file_path, sep='\t', names=["col%i" % i for i in range(9)])
        if self.is_train:
            label_dict = self.get_label_dict()
            df["label"] = df["col8"].map(label_dict)
        return df

    def get_sent_with_entity_token(self, idx):
        row = self.data.loc[idx]
        sent = row["col1"]
        sent = list(sent)
        if row["col3"] < row["col6"]:
            sent.insert(self.data.loc[idx, "col7"] + 1, " [/ENT] ")
            sent.insert(self.data.loc[idx, "col6"], " [ENT] ")
            sent.insert(self.data.loc[idx, "col4"] + 1, " [/ENT] ")
            sent.insert(self.data.loc[idx, "col3"], " [ENT] ")
        else:
            sent.insert(self.data.loc[idx, "col4"] + 1, " [/ENT] ")
            sent.insert(self.data.loc[idx, "col3"], " [ENT] ")
            sent.insert(self.data.loc[idx, "col7"] + 1, " [/ENT] ")
            sent.insert(self.data.loc[idx, "col6"], " [ENT] ")
        return "".join(sent)

    def get_label_dict(self):
        with open(self.label_dict_file, "rb") as f:
            return pickle.load(f)

    def get_entity_token_indices(self, input_ids: torch.tensor):
        entity_open_token_id = self.tokenizer.convert_tokens_to_ids("[ENT]")
        return torch.where(input_ids == entity_open_token_id)[0]


class EntityDatasetV2(EntityDataset):
    def __init__(self, file_path, tokenizer, max_length=512, is_train=True):
        super().__init__(file_path, tokenizer, max_length, is_train)

    def __getitem__(self, index):
        inputs = super().__getitem__(index)
        inputs.update({"entity_ids": self.get_entity_ids(inputs["input_ids"])})
        return inputs

    def get_entity_ids(self, input_ids):
        entity_ids = [0] * len(input_ids)
        entity_open_token_id = self.tokenizer.convert_tokens_to_ids("[ENT]")
        entity_close_token_id = self.tokenizer.convert_tokens_to_ids("[/ENT]")
        entity_open_token_indices = torch.where(input_ids == entity_open_token_id)[0]
        entity_close_token_indices = torch.where(input_ids == entity_close_token_id)[0]
        for op, cl in zip(entity_open_token_indices.tolist(),
                          entity_close_token_indices.tolist()):
            for i in range(op, cl + 1):
                entity_ids[i] = 1
        return torch.tensor(entity_ids)


class EntityDatasetV3(EntityDataset):
    def __init__(self, file_path, tokenizer, max_length=512, is_train=True):
        super().__init__(file_path, tokenizer, max_length, is_train)

    def get_sent_with_entity_token(self, idx):
        row = self.data.loc[idx]
        sent = row["col1"]
        sent = list(sent)
        if row["col3"] < row["col6"]:
            sent.insert(self.data.loc[idx, "col7"] + 1, " [/E2] ")
            sent.insert(self.data.loc[idx, "col6"], " [E2] ")
            sent.insert(self.data.loc[idx, "col4"] + 1, " [/E1] ")
            sent.insert(self.data.loc[idx, "col3"], " [E1] ")
        else:
            sent.insert(self.data.loc[idx, "col4"] + 1, " [/E1] ")
            sent.insert(self.data.loc[idx, "col3"], " [E1] ")
            sent.insert(self.data.loc[idx, "col7"] + 1, " [/E2] ")
            sent.insert(self.data.loc[idx, "col6"], " [E2] ")
        return "".join(sent)

    def get_entity_token_indices(self, input_ids: torch.tensor):
        entity_token1_id = self.tokenizer.convert_tokens_to_ids("[E1]")
        entity_token2_id = self.tokenizer.convert_tokens_to_ids("[E2]")
        idx1 = torch.where(input_ids == entity_token1_id)[0]
        idx2 = torch.where(input_ids == entity_token2_id)[0]
        return torch.cat((idx1, idx2))


def get_k_fold_dataset(dataset: EntityDataset, k=5, random_state=1818):
    k_fold = KFold(k, shuffle=True, random_state=random_state)
    dummy_x = [0] * dataset.data["col0"].nunique()
    unique_ref = dataset.data["col0"].unique()
    idx = 0
    for train_idx, val_idx in k_fold.split(dummy_x):
        train_ref = unique_ref[train_idx]
        train_idx = dataset.data[dataset.data["col0"].isin(train_ref)].index
        val_ref = unique_ref[val_idx]
        val_idx = dataset.data[dataset.data["col0"].isin(val_ref)].index
        yield idx, Subset(dataset, train_idx), Subset(dataset, val_idx)
        idx += 1
