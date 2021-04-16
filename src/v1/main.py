import sys

sys.path.append("/opt/ml/project")

import torch
from torch.utils.data import DataLoader

from src.v1 import *
from utils import fix_random_state

if __name__ == '__main__':
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
    data_loader = DataLoader(dataset,
                             batch_size=PARAMS["config"]["batch_size"],
                             shuffle=True,
                             num_workers=2,
                             pin_memory=True)
    train(model, optimizer, criterion, PARAMS["config"]["epoch"], data_loader)
    test_dataset = RelationDataset(tokenizer, PARAMS["config"]["max_length"], False)
    test_loader = DataLoader(test_dataset,
                             batch_size=PARAMS["config"]["batch_size"],
                             shuffle=False,
                             num_workers=2)
    inference(model, test_loader)
