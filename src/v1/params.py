PARAMS = {
    "project": "relation-extraction",
    "group": "bert-multilingual",
    "job_type": "v1",
    "config": {
        "model": "bert-base-multilingual-cased",
        "epoch": 20,
        "batch_size": 40,
        "lr": 1e-3,
        "f-lr": 1e-4,
        "criterion": "ce",
        "optimizer": "adam",
        "max_length": 400
    }
}
