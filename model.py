import torch
import torch.nn as nn
from gluonnlp.data import SentencepieceTokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertEmbeddings, BertModel
from transformers.models.electra.modeling_electra import ElectraEmbeddings, ElectraModel

from utils import fix_random_state


def get_model_and_tokenizer(config):
    fix_random_state()
    model_name = config["model"]
    add_entity_embeddings = config["add_entity_embeddings"]
    tokenizer_version = config["tokenizer"]
    if config["type"] == "transformers":
        backbone = AutoModel.from_pretrained(model_name)
        if "electra" in model_name and add_entity_embeddings:
            backbone = ElectraRelationModel(backbone)
        if "bert" in model_name and add_entity_embeddings:
            backbone = BertRelationModel(backbone)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer_version == "v1":
            tokenizer.add_special_tokens({"additional_special_tokens": ["[ENT]", "[/ENT]"]})
        elif tokenizer_version == "v2":
            tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})
        else:
            raise ValueError("incorrect tokenizer version: %s" % tokenizer_version)
    elif model_name == "KoBERT":
        backbone, vocab = get_pytorch_kobert_model()
        if add_entity_embeddings:
            backbone = BertRelationModel(backbone)
        tok_path = get_tokenizer()
        spt = SentencepieceTokenizer(tok_path)
        tokenizer = get_kobert_tokenizer(tokenizer_version, vocab, spt,
                                         config["max_length"])
    else:
        raise ValueError("incorrect model_name: %s" % model_name)
    backbone.resize_token_embeddings(len(tokenizer))
    return RelationModel(backbone), tokenizer


def get_optimizer(model, config):
    f_lr = config["f-lr"]
    lr = config["lr"]
    optimizer = torch.optim.Adam([
        {"params": [param for name, param in model.backbone.named_parameters()
                    if "relation_embeddings" not in name],
         "lr": f_lr},
        {"params": [param for name, param in model.backbone.named_parameters()
                    if "relation_embeddings" in name],
         "lr": lr},
        {"params": model.fc.parameters(), "lr": lr}
    ], lr=0)
    return optimizer


# legacy
def add_word_embedding_vector(model, num=2):
    model.embeddings.word_embeddings.weight = nn.Parameter(
        torch.cat((model.embeddings.word_embeddings.weight,
                   torch.randn(num, model.embeddings.word_embeddings.weight.size(1))))
    )
    model.embeddings.word_embeddings.num_embeddings += num
    return model


def get_kobert_tokenizer(version, vocab, spt, max_length=400):
    if version == "v1":
        return Tokenizer(vocab, spt, max_length)
    if version == "v2":
        return TokenizerV2(vocab, spt, max_length)
    raise ValueError("incorrect kobert tokenizer version: %s" % version)


class Tokenizer:
    def __init__(self, vocab, spt, max_length=400):
        self.vocab = vocab
        self.spt = spt
        self.max_length = max_length
        _entity_tokens = ["[ENT]", "[/ENT]"]
        self.entity_tokens = {token: len(vocab) + i
                              for i, token in enumerate(_entity_tokens)}
        self.idx_to_token = self.vocab.idx_to_token
        self.idx_to_token.extend(_entity_tokens)

    def __len__(self):
        return len(self.idx_to_token)

    def __call__(self, sent, **kwargs):
        padding_id = self.vocab("[PAD]")
        input_ids = self.encode(sent)
        attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
        input_ids.extend([padding_id] * (self.max_length - len(input_ids)))
        token_type_ids = [0] * self.max_length
        inputs = {"input_ids": input_ids,
                  "token_type_ids": token_type_ids,
                  "attention_mask": attention_mask}
        if kwargs.get("return_tensors", None) == "pt":
            return {key: torch.tensor(val) for key, val in inputs.items()}
        return inputs

    def encode(self, sent):
        input_ids = [self.vocab("[CLS]")]
        for s in sent.split():
            if s in self.entity_tokens:
                input_ids.append(self.entity_tokens[s])
            else:
                input_ids.extend(self.vocab(self.spt(s)))
        input_ids.append(self.vocab("[SEP]"))
        return input_ids

    def convert_tokens_to_ids(self, token):
        if token in self.entity_tokens:
            return self.entity_tokens[token]
        return self.vocab(token)

    def decode(self, input_ids: [int, list]):
        padding_id = self.vocab("[PAD]")
        if isinstance(input_ids, (list, tuple)):
            tokens = []
            for input_id in input_ids:
                if input_id == padding_id:
                    break
                else:
                    tokens.append(self.idx_to_token[input_id])
            return "".join(tokens).replace("▁", " ").strip()
        return self.idx_to_token[input_ids]


class TokenizerV2(Tokenizer):
    def __init__(self, vocab, spt, max_length=400):
        super().__init__(vocab, spt, max_length)
        _entity_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
        self.entity_tokens = {token: len(vocab) + i
                              for i, token in enumerate(_entity_tokens)}
        self.idx_to_token = self.vocab.idx_to_token
        self.idx_to_token.extend(_entity_tokens)


class BertRelationEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.relation_embeddings = nn.Embedding(2, config.hidden_size)

    def forward(self,
                input_ids=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None,
                past_key_values_length=0, entity_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if entity_ids is not None:
            embeddings += self.relation_embeddings(entity_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertRelationModel(BertModel):
    def __init__(self, model: BertModel):
        super().__init__(model.config)
        self.embeddings = BertRelationEmbeddings(model.config)
        self.embeddings.word_embeddings = model.embeddings.word_embeddings
        self.embeddings.position_embeddings = model.embeddings.position_embeddings
        self.embeddings.token_type_embeddings = model.embeddings.token_type_embeddings
        self.encoder = model.encoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            entity_ids=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            entity_ids=entity_ids
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class ElectraRelationEmbeddings(ElectraEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.relation_embeddings = nn.Embedding(2, config.embedding_size)
        self.relation_embeddings.weight = nn.Parameter(self.relation_embeddings.weight / 100)

    def forward(self,
                input_ids=None, token_type_ids=None,
                position_ids=None, inputs_embeds=None,
                past_key_values_length=0, entity_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        if entity_ids is not None:
            embeddings += self.relation_embeddings(entity_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ElectraRelationModel(ElectraModel):
    def __init__(self, model: ElectraModel):
        super().__init__(model.config)
        self.embeddings = ElectraRelationEmbeddings(model.config)
        self.embeddings.word_embeddings = model.embeddings.word_embeddings
        self.embeddings.position_embeddings = model.embeddings.position_embeddings
        self.embeddings.token_type_embeddings = model.embeddings.token_type_embeddings
        self.encoder = model.encoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                entity_ids=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            entity_ids=entity_ids
        )

        if hasattr(self, "embeddings_project"):
            hidden_states = self.embeddings_project(hidden_states)

        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return hidden_states


class RelationModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.fc = nn.Linear(hidden_size * 2, 42)

    def forward(self, *args, **kwargs):
        entity_token_indices = kwargs.pop("entity_token_indices", None)
        result = self.backbone(*args, **kwargs)
        outputs = result.last_hidden_state
        outputs = self.get_relation_output(outputs, entity_token_indices)
        return self.fc(outputs)

    def get_relation_output(self, outputs, entity_token_indices):
        entity_vectors = []
        for i in range(entity_token_indices.size(0)):
            output = torch.index_select(outputs[i], 0, entity_token_indices[i]).reshape(1, -1)
            entity_vectors.append(output)
        return torch.cat(entity_vectors)
