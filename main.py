import sys

sys.path.append("/opt/ml/project")

import fire
import wandb
import torch
from torch.utils.data import DataLoader

from model import get_model_and_tokenizer, get_optimizer
from dataset import RelationDataset, k_fold_dataset
from loss import get_criterion
from train import train
from params import PARAMS
from inference import inference


def main(mode="test", k_fold=False):
    model, tokenizer = get_model_and_tokenizer(PARAMS["config"]["model"],
                                               PARAMS["config"]["add_relation_embeddings"])
    optimizer = get_optimizer(model)
    criterion = get_criterion(PARAMS["config"]["criterion"])
    dataset = RelationDataset(tokenizer, PARAMS["config"]["max_length"])
    if mode == "test":
        print("test start!")
        wandb.login()
        for k_idx, train_dataset, val_dataset in k_fold_dataset(dataset):
            model.cuda()
            PARAMS.update({"name": "kfold-%s" % k_idx})
            train_loader = DataLoader(train_dataset,
                                      batch_size=PARAMS["config"]["batch_size"],
                                      shuffle=True,
                                      num_workers=PARAMS["config"]["num_workers"],
                                      pin_memory=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=PARAMS["config"]["batch_size"],
                                    shuffle=True,
                                    num_workers=2,
                                    pin_memory=True)
            with wandb.init(**PARAMS) as logger:
                train(model, optimizer, criterion, PARAMS["config"]["epoch"],
                      train_loader, val_loader, logger)
            torch.cuda.empty_cache()
            if not k_fold:
                break
            model, tokenizer = get_model_and_tokenizer(PARAMS["config"]["model"])
            optimizer = get_optimizer(model)
    elif mode == "inference":
        print("inference start!")
        model.cuda()
        data_loader = DataLoader(dataset,
                                 batch_size=PARAMS["config"]["batch_size"],
                                 shuffle=True,
                                 num_workers=PARAMS["config"]["num_workers"],
                                 pin_memory=True)
        train(model, optimizer, criterion, PARAMS["config"]["epoch"], data_loader)
        test_dataset = RelationDataset(tokenizer, PARAMS["config"]["max_length"], False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=PARAMS["config"]["batch_size"],
                                 shuffle=False,
                                 num_workers=2)
        inference(model, test_loader)
    else:
        raise ValueError("incorrect mode: %s" % mode)
    return


if __name__ == '__main__':
    fire.Fire(main)
