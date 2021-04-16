import sys

sys.path.append("/opt/ml/project")

import wandb
import torch
from torch.utils.data import DataLoader

from src.v1 import *
from utils import fix_random_state

if __name__ == '__main__':
    wandb.login()
    fix_random_state()
    tokenizer = get_tokenizer(PARAMS["config"]["model"])
    dataset = RelationDataset(tokenizer, PARAMS["config"]["max_length"])
    model = RelationModel(PARAMS["config"]["model"])
    model.cuda()
    criterion = get_criterion(PARAMS["config"]["criterion"])
    optimizer = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": PARAMS["config"]["f-lr"]},
        {"params": model.fc.parameters(), "lr": PARAMS["config"]["lr"]}
    ], lr=0)
    for k_idx, train_dataset, val_dataset in k_fold_dataset(dataset):
        PARAMS.update({"name": "kfold-%s" % k_idx})
        train_loader = DataLoader(train_dataset,
                                  batch_size=PARAMS["config"]["batch_size"],
                                  shuffle=True,
                                  num_workers=2,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=PARAMS["config"]["batch_size"],
                                shuffle=True,
                                num_workers=1,
                                pin_memory=True)
        with wandb.init(**PARAMS) as logger:
            train(model, optimizer, criterion, PARAMS["config"]["epoch"],
                  train_loader, val_loader, logger)
        torch.cuda.empty_cache()
        break
