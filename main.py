import sys

sys.path.append("/opt/ml/project")

import fire
import wandb
import torch
from torch.utils.data import DataLoader

from model import get_model_and_tokenizer, get_optimizer
from dataset import get_entity_dataset, get_k_fold_dataset
from loss import get_criterion
from train import train
from params import PARAMS
from inference import inference
from slack import alert


@alert
def main(mode="test", k_fold=False):
    model, tokenizer = get_model_and_tokenizer(PARAMS["config"])
    optimizer = get_optimizer(model, PARAMS["config"])
    criterion = get_criterion(PARAMS["config"]["criterion"])
    dataset = get_entity_dataset(PARAMS["config"]["dataset_version"],
                                 PARAMS["config"]["train_file"],
                                 tokenizer,
                                 max_length=PARAMS["config"]["max_length"])
    batch_size = PARAMS["config"]["batch_size"]
    num_workers = PARAMS["config"]["num_workers"]
    if mode == "test":
        print("test start!")
        wandb.login()
        for k_idx, train_dataset, val_dataset in get_k_fold_dataset(dataset, PARAMS["config"]["k_fold"]):
            model.cuda()
            PARAMS.update({"name": "kfold-%s" % k_idx})
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      pin_memory=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=True)
            with wandb.init(reinit=True, **PARAMS) as logger:
                train(model, optimizer, criterion, PARAMS["config"]["epoch"],
                      train_loader, val_loader, logger)
            torch.cuda.empty_cache()
            if not k_fold:
                break
            model, tokenizer = get_model_and_tokenizer(PARAMS["config"])
            optimizer = get_optimizer(model, PARAMS["config"])
    elif mode == "inference":
        print("inference start!")
        model.cuda()
        data_loader = DataLoader(dataset,
                                 batch_size=PARAMS["config"]["batch_size"],
                                 shuffle=True,
                                 num_workers=PARAMS["config"]["num_workers"],
                                 pin_memory=True)
        train(model, optimizer, criterion, PARAMS["config"]["epoch"], data_loader)
        test_file = "/opt/ml/input/data/test/test.tsv"
        test_dataset = get_entity_dataset(PARAMS["config"]["dataset_version"],
                                          test_file,
                                          tokenizer,
                                          max_length=PARAMS["config"]["max_length"],
                                          is_train=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=PARAMS["config"]["batch_size"],
                                 shuffle=False,
                                 num_workers=PARAMS["config"]["num_workers"])
        inference(model, test_loader)
        model_name = PARAMS["config"]["model"].replace("/", "-")
        model_name = "-".join([model_name, PARAMS["job_type"]])
        torch.save(model.state_dict(), "/opt/ml/project/result/models/%s.pth" % model_name)
    else:
        raise ValueError("incorrect mode: %s" % mode)
    return


if __name__ == '__main__':
    fire.Fire(main)
