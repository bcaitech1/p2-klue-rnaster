import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def get_model(model_name):
    model = AutoModel.from_pretrained(model_name)
    model.embeddings.word_embeddings.weight = nn.Parameter(
        torch.cat((model.embeddings.word_embeddings.weight,
                   torch.randn(2, model.embeddings.word_embeddings.weight.size(1))))
    )
    model.embeddings.word_embeddings.num_embeddings += 2
    return model


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT]", "[/ENT]"]})
    return tokenizer


class RelationModel(nn.Module):
    def __init__(self, model_name):
        super(RelationModel, self).__init__()
        self.backbone = get_model(model_name)
        self.fc = nn.Linear(768 * 2, 42)

    def forward(self, *args, **kwargs):
        entity_indices = kwargs["entity_indices"]
        del kwargs["entity_indices"]
        result = self.backbone(*args, **kwargs)
        outputs = result.last_hidden_state
        outputs = self.get_relation_output(outputs, entity_indices)
        return self.fc(outputs)

    def get_relation_output(self, outputs, entity_indices):
        entity_vectors = []
        for i in range(entity_indices.size(0)):
            output = torch.index_select(outputs[i], 0, entity_indices[i]).reshape(1, -1)
            entity_vectors.append(output)
        return torch.cat(entity_vectors)
