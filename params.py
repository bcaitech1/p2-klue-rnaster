PARAMS = {
    "project": "relation-extraction",
    "group": "KoElectra",
    "job_type": "v12",
    "config": {
        "type": "transformers",
        "model": "monologg/koelectra-base-v3-discriminator",
        "tokenizer": "v2",
        "dataset_version": "v3",
        "add_entity_embeddings": False,
        "epoch": 20,
        "batch_size": 40,
        "lr": 3e-6,
        "f-lr": 3e-7,
        "criterion": "ce",
        "optimizer": "adam",
        "max_length": 400,
        "num_workers": 1,
        "k_fold": 5,
        "available-models": ["bert-base-multilingual-cased",
                             "KoBERT",
                             "monologg/koelectra-base-v3-discriminator"],
        # "comments": ["[ENT], [/ENT] -> [E1], [/E1], [E2], [/E2]"],
        "train_file": "/opt/ml/input/data/train/train.tsv"
    }
}
