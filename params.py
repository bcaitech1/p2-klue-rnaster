PARAMS = {
    "project": "relation-extraction",
    "group": "KoBERT",
    "job_type": "v1-2",
    "config": {
        "model": "KoBERT",
        "epoch": 5,
        "batch_size": 40,
        "lr": 3e-4,
        "f-lr": 3e-5,
        "criterion": "ce",
        "optimizer": "adam",
        "max_length": 400,
        "num_workers": 4,
        "available-models": ["bert-base-multilingual-cased", "KoBERT"]
    }
}
