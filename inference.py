from pathlib import Path

import pandas as pd
import torch

from params import PARAMS


@torch.no_grad()
def inference(model, data_loader):
    save_path = "/opt/ml/project/result"
    file_name = PARAMS["config"]["model"].replace("/", "-")
    result = []
    Path(save_path).mkdir(parents=True, exist_ok=True)
    model.eval()
    for inputs in data_loader:
        inputs = {key: val.cuda() for key, val in inputs.items()}
        inputs.pop("labels")
        outputs = model(**inputs)
        pred = outputs.argmax(1).detach().tolist()
        result.extend(pred)
    pd.DataFrame({"pred": result}) \
        .to_csv(save_path + "/%s-%s.csv" % (file_name, PARAMS["job_type"]), index=False)
    return
