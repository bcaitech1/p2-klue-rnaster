import torch
from sklearn.metrics import f1_score, accuracy_score

from utils import send_inputs_to_gpu


def train(model, optimizer, criterion, epoch, train_loader,
          val_loader=None, logger=None):
    step = 1
    for i in range(epoch):
        for inputs in train_loader:
            model.train()
            optimizer.zero_grad()
            (input_ids, token_type_ids, attention_mask,
             entity_indices, labels) = send_inputs_to_gpu(inputs)
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            entity_indices=entity_indices)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if logger is not None and step % 20 == 0:
                train_acc, train_f1 = get_accuracy_and_f1(outputs, labels)
                val_loss, val_acc, val_f1 = evaluate(model, criterion, val_loader)
                logger.log({
                    "epoch": i,
                    "train_loss": loss.item(),
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1
                }, step=step)
            step += 1
    return


def get_accuracy_and_f1(outputs, labels):
    _outputs = outputs.detach().argmax(1).tolist()
    _labels = labels.detach().tolist()
    return (accuracy_score(_outputs, _labels, ),
            f1_score(_outputs, _labels, average="macro"))


@torch.no_grad()
def evaluate(model, criterion, val_loader):
    val_loss = 0
    val_acc = 0
    val_f1 = 0
    step = 0
    model.eval()
    for inputs in val_loader:
        (input_ids, token_type_ids, attention_mask,
         entity_indices, labels) = send_inputs_to_gpu(inputs)
        outputs = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        entity_indices=entity_indices)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        acc, f1 = get_accuracy_and_f1(outputs, labels)
        val_acc += acc
        val_f1 += f1
        step += 1
    return val_loss / step, val_acc / step, val_f1 / step
