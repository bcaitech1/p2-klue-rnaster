import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer

from params import PARAMS
from utils import fix_random_state


def get_model_and_tokenizer(model_name):
    fix_random_state()
    if model_name == "bert-base-multilingual-cased":
        backbone = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT]", "[/ENT]"]})
    elif model_name == "KoBERT":
        backbone, vocab = get_pytorch_kobert_model()
        tok_path = get_tokenizer()
        spt = SentencepieceTokenizer(tok_path)
        tokenizer = Tokenizer(vocab, spt, max_length=PARAMS["config"]["max_length"])
    else:
        raise ValueError("incorrect model_name: %s" % model_name)
    backbone = add_word_embedding_vector(backbone)
    return RelationModel(backbone), tokenizer


def add_word_embedding_vector(model, num=2):
    model.embeddings.word_embeddings.weight = nn.Parameter(
        torch.cat((model.embeddings.word_embeddings.weight,
                   torch.randn(num, model.embeddings.word_embeddings.weight.size(1))))
    )
    model.embeddings.word_embeddings.num_embeddings += num
    return model


class Tokenizer:
    def __init__(self, vocab, spt, max_length=400):
        self.vocab = vocab
        self.spt = spt
        self.max_length = max_length
        self.entity_tokens = ["[ENT]", "[/ENT]"]
        self.open_entity_id = len(vocab) + 1
        self.closed_entity_id = len(vocab) + 2

    def __len__(self):
        return len(self.vocab) + 2

    def __call__(self, sent, **kwargs):
        input_ids = self.encode(sent)
        attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        input_ids.extend([1] * (self.max_length - len(input_ids)))
        token_type_ids = [0] * self.max_length
        return {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask}

    def encode(self, sent):
        input_ids = [self.vocab("[CLS]")]
        for s in sent.split():
            if s == "[ENT]":
                input_ids.append(8002)
            elif s == "[/ENT]":
                input_ids.append(8003)
            else:
                input_ids.extend(self.vocab(self.spt(s)))
        input_ids.append(self.vocab("[SEP]"))
        return input_ids

    def convert_tokens_to_ids(self, token):
        if token == "[ENT]":
            return self.open_entity_id
        if token == "[/ENT]":
            return self.closed_entity_id
        return self.vocab(token)

    def decode(self, input_ids: [int, list]):
        if isinstance(input_ids, (list, tuple)):
            tokens = []
            for input_id in input_ids:
                if input_id == 1: break
                if input_id == 8002:
                    tokens.append(" [ENT]")
                elif input_id == 8003:
                    tokens.append(" [/ENT]")
                else:
                    tokens.append(self.vocab.idx_to_token[input_id])
            return "".join(tokens).replace("▁", " ").strip()
        return self.vocab.idx_to_token[input_ids]


class RelationModel(nn.Module):
    def __init__(self, backbone):
        super(RelationModel, self).__init__()
        self.backbone = backbone
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
