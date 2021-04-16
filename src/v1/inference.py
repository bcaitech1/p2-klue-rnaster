from pathlib import Path

import pandas as pd
import torch

from utils import send_inputs_to_gpu
from src.v1 import PARAMS


@torch.no_grad()
def inference(model, data_loader):
    save_path = "/opt/ml/project/src/v1/result"
    result = []
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.eval()
    for inputs in data_loader:
        (input_ids, token_type_ids, attention_mask,
         entity_indices, labels) = send_inputs_to_gpu(inputs)
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        entity_indices=entity_indices)
        pred = outputs.argmax(1).detach().tolist()
        result.extend(pred)
    pd.DataFrame({"pred": result}) \
        .to_csv(save_path + "/%s-%s.csv" % (PARAMS["config"]["model"], PARAMS["job_type"]), index=False)
    return
