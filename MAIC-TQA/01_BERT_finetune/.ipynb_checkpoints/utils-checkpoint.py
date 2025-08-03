import torch
import numpy as np
from transformers import BertTokenizer
import json
import pandas as pd
from torch import nn
from transformers import BertModel
from torch.optim import Adam
from tqdm import tqdm
from ipywidgets import FloatProgress
from torch.utils.tensorboard import SummaryWriter
import re
from seqeval.metrics import classification_report
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import json
import pandas as pd

def json2dataframe(all_datas, tokenizer, slots_num, intents_num):
    df = pd.DataFrame(columns=['category', 'text', 'intents', 'slots'])
    intents = []
    for data in all_datas:
        query = data['query']
        if type(data['slots']) == type(''):
            slots = data['slots'].split(' ')
        else:
            slots = data['slots']
        encoded_text = tokenizer(query, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
        slots = data['token_slots']
        numbered_slots = [slots_num[item] for item in slots]
        intents_label = [0.0] * len(intents_num)
        if data['intent'] not in intents and '+' not in data['intent']:
            intents.append(data['intent'])
        if '+' in data['intent']:
            data['intent'] = data['intent'].split('+')
            for intent in data['intent']:
                intents_label[intents_num[intent]] = 1.0
            df = pd.concat([df, pd.DataFrame([{'category': intents_label, 'text': data['query'], 'intents': 1, 'slots': numbered_slots}])], ignore_index=True)
        elif '+' not in data['intent']:
            intent = data['intent']
            intents_label[intents_num[intent]] = 1.0
            df = pd.concat([df, pd.DataFrame([{'category': intents_label, 'text': data['query'], 'intents': 0, 'slots': numbered_slots}])], ignore_index=True)
    df['slots'] = df['slots'].apply(pad_to_512)
    return df



def pad_to_512(input_string, max_pad_lenth=512):
    while len(input_string) < max_pad_lenth:
        input_string.append(int(-100))
    return input_string


def tensors_equal_ignore_order(tensor1, tensor2):
    # Sort the two tensors along the specified dimension
    sorted_tensor1, _ = torch.sort(tensor1)
    sorted_tensor2, _ = torch.sort(tensor2)
    results = []
    for row1, row2 in zip(sorted_tensor1, sorted_tensor2):
        results.append(torch.equal(row1, row2))
        results_tensor = torch.tensor(results, dtype=torch.bool)
    return results_tensor

# Input two tensors: the first tensor is the probability tensor, and the second tensor is the one-hot encoded label tensor.
def compute_multi_label_acc(probility, label):
    probility, idx1 = torch.sort(probility, descending=True)
    label, idx2 = torch.sort(label, descending=True)
    idx1 = idx1[:,0:2]
    idx2 = idx2[:,0:2]
    for i,labl in enumerate(label):
        if labl.sum() < 2:
            idx1[i,1] = 0
            idx2[i,1] = 0
    acc = tensors_equal_ignore_order(idx1, idx2).sum().item()
    return acc



def train(model, train_data, val_data, learning_rate, epochs, writer, tokenizer, batch_size):
    # Sort the two tensors along the specified dimension using the Dataset class to retrieve the training and validation sets
    train, val = Dataset(train_data, tokenizer), Dataset(val_data, tokenizer)
    # Use DataLoader to retrieve data based on batch_size, and shuffle samples during training
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # loss
    criterion = nn.CrossEntropyLoss()
    binary_criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    for epoch_num in range(epochs):
        total_intent_acc_train = 0
        total_num_acc_train = 0
        total_loss_train = 0
        total_slot_acc_train = 0
        total_tokens_train = 0
        for train_input, train_intent_label, train_num_label, train_slot_label in tqdm(train_dataloader):
            train_intent_label = train_intent_label.to(device)
            train_num_label = train_num_label.to(device)
            train_slot_label = train_slot_label.to(device)
            mask = train_input['attention_mask'].squeeze(1).to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # model output
            intent_probability, num_intents, slot_probability = model(input_id, mask)
            # compute loss
            intent_loss = binary_criterion(intent_probability, train_intent_label.float())
            active_loss = mask.view(-1) == 1
            active_logits = slot_probability.view(-1, 13)[active_loss]
            active_labels = train_slot_label.view(-1)[active_loss]
            slot_loss = criterion(active_logits, active_labels)
            num_loss = criterion(num_intents, train_num_label)
            loss = intent_loss + num_loss + slot_loss
            total_loss_train += loss.item()
            # compute metric
            intent_acc = compute_multi_label_acc(intent_probability, train_intent_label)
            total_intent_acc_train += intent_acc
            num_intent_acc = (num_intents.argmax(dim=1) == train_num_label).sum().item()
            total_num_acc_train += num_intent_acc
            word_leval_slots_acc = (slot_probability.argmax(dim=2).view(-1)[active_loss] == train_slot_label.view(-1)[active_loss]).sum().item()
            batch_token_nums = active_loss.sum().item()
            total_slot_acc_train += word_leval_slots_acc
            total_tokens_train += batch_token_nums
            model.zero_grad()
            loss.backward()
            optimizer.step()
        # ------- val -----------
        total_intent_acc_val = 0
        total_num_acc_val = 0
        total_loss_val = 0
        total_slot_acc_val = 0
        total_tokens_val = 0
        with torch.no_grad():
            for val_input, val_intent_label, val_num_label, val_slot_label in val_dataloader:
                val_intent_label = val_intent_label.to(device)
                val_num_label = val_num_label.to(device)
                val_slot_label = val_slot_label.to(device)
                mask = val_input['attention_mask'].squeeze(1).to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)
                intent_probability, num_intents, slot_probability = model(input_id, mask)
                intent_loss = binary_criterion(intent_probability, val_intent_label.float())
                # compute slots loss
                active_loss = mask.view(-1) == 1
                active_logits = slot_probability.view(-1, 13)[active_loss]
                active_labels = val_slot_label.view(-1)[active_loss]
                slot_loss = criterion(active_logits, active_labels)
                # compute intent num loss
                num_loss = criterion(num_intents, val_num_label)
                loss = intent_loss + num_loss + slot_loss
                total_loss_val += loss.item()
                # compute metric
                intent_acc = compute_multi_label_acc(intent_probability, val_intent_label)
                total_intent_acc_val += intent_acc
                num_intent_acc = (num_intents.argmax(dim=1) == val_num_label).sum().item()
                total_num_acc_val += num_intent_acc
                word_leval_slots_acc = (slot_probability.argmax(dim=2).view(-1)[active_loss] == val_slot_label.view(-1)[active_loss]).sum().item()
                batch_token_nums = active_loss.sum().item()
                total_slot_acc_val += word_leval_slots_acc
                total_tokens_val += batch_token_nums
        writer.add_scalar('Loss/train', total_loss_train / len(train_data), epoch_num)
        writer.add_scalar('Accuracy/train_intent', total_intent_acc_train / len(train_data), epoch_num)
        writer.add_scalar('Accuracy/train_num_intents', total_num_acc_train / len(train_data), epoch_num)
        writer.add_scalar('Accuracy/train_token_level_slot_acc', total_slot_acc_train / total_tokens_train, epoch_num)
        writer.add_scalar('Loss/val', total_loss_val / len(val_data), epoch_num)
        writer.add_scalar('Accuracy/val_intent', total_intent_acc_val / len(val_data), epoch_num)
        writer.add_scalar('Accuracy/val_num_intents', total_num_acc_val / len(val_data), epoch_num)
        writer.add_scalar('Accuracy/val_token_level_slot_acc', total_slot_acc_val / total_tokens_val, epoch_num)

        print(
            f'''Epochs: {epoch_num + 1} 
            | Train Loss: {total_loss_train / len(train_data): .3f} 
            | Train Intent Accuracy: {total_intent_acc_train / len(train_data): .3f}
            | Train Num of intents Accuracy: {total_num_acc_train / len(train_data): .3f} 
            | Train Token-level Slots Accuracy: {total_slot_acc_train / total_tokens_train: .3f} 
            | Val Loss: {total_loss_val / len(val_data): .3f} 
            | Val Intent Accuracy: {total_intent_acc_val / len(val_data): .3f}
            | Val Num of intents Accuracy: {total_num_acc_val / len(val_data): .3f}
            | Val Token-level Slots Accuracy: {total_slot_acc_val / total_tokens_val: .3f} ''')
        writer.close()



def evaluate(model, datas, slots_num, all_intents, tokenizer, device):
    # Load data
    # Initialize storage variables
    pred_intent_num = []
    true_intent_num = []
    pred_intent_label = []
    true_intent_label = []
    true_key_words = []
    pred_key_words = []
    true_token_slots = []
    pred_token_slots = []
    
    # Iterate through the data
    for data in tqdm(datas):
        query = data['query']
        if type(data['slots']) == type(''):
            origin_slots = data['slots'].split(' ')
        else:
            origin_slots = data['slots']
        intent = data['intent']
        intent_label = intent2label(intent)
        true_intent_label.append(list(intent_label))
        # Text encoding
        encoded_text = tokenizer(query, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
        new_tokens, new_slots = align_tokens_with_query(tokens, query, origin_slots)
        # True slot
        tokens_slots = data['token_slots']
        true_token_slots.append(tokens_slots)
        # Model prediction
        encoded_text_id = encoded_text['input_ids'].to(device)
        mask = encoded_text['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(encoded_text_id, mask)
        # Intent prediction
        intent_probility = outputs[0].view(-1)
        _, intent_idx = torch.topk(intent_probility, k=2, dim=0)
        intent_idx = intent_idx.cpu()
        intent_num_probility = outputs[1].argmax()
        pred_intent_num.append(intent_num_probility.cpu())
        if intent_num_probility == 0:
            intent_idx = intent_idx[0]
        pred_intent = [1 if i in intent_idx else 0 for i in range(15)]
        if isinstance(intent_idx, torch.Tensor):
            intent_idx = intent_idx.flatten().tolist()
        intent = [all_intents[idx] for idx in intent_idx]
        data["BERT_pred_intent"] = intent
        pred_intent_label.append(list(pred_intent))
        # Slot prediction
        slots_probility = outputs[2].argmax(dim=2).view(-1)
        token_slot = [find_key(slots_num, i)[0] for i in slots_probility]
        pred_key_word = restore_keywords_from_tokens(new_tokens, token_slot)
        key_word = pred_key_word
        data["BERT_pred_slots"] = key_word
        pred_slots_num = slots_probility.cpu()
        pred_slots_num = [t.item() for t in pred_slots_num]
        pred_token_slots.append(num_2_slots(pred_slots_num, slots_num))

    # Compute classification report
    print('### Intent Report ###:')
    inetnt_report = sk_classification_report(true_intent_label, pred_intent_label, digits=4)
    print(inetnt_report)
    
    print('### Slots Report ###:')
    report = classification_report(true_token_slots, pred_token_slots, digits=4)
    print(report)
    
    return datas



def restore_keywords_from_query(query, slots):
    keywords = []
    current_tokens = []
    current_label = None
    query = list(query)
    if slots[0] == '[CLS]':
        slots = slots[1:-1]

    for token, slot in zip(query, slots):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))

    return keywords



def num_2_slots(num_list, slots_num):
    # Create a reverse mapping dictionary
    num_to_slot = {v: k for k, v in slots_num.items()}
    # Convert numbers to corresponding labels
    return [num_to_slot[num] for num in num_list]


def restore_keywords_from_tokens(tokens, token_slot):
    keywords = []
    current_tokens = []
    current_label = None
    token_slot = token_slot[1:-1]

    for token, slot in zip(tokens, token_slot):
        if slot.startswith('B-'):
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
            current_label = slot[2:]
            current_tokens.append(token)
        elif slot.startswith('I-') and current_label == slot[2:]:
            current_tokens.append(token)
        else:
            if current_tokens:
                keywords.append((''.join(current_tokens), current_label))
                current_tokens = []
                current_label = None

    if current_tokens:
        keywords.append((''.join(current_tokens), current_label))
    keyword_pair = []
    for keyword in keywords:
        if keyword[-1] == 'city':
            keyword_pair.append(f'city:{keyword[0]}')
        elif keyword[-1] == 'district':
            keyword_pair.append(f'district:{keyword[0]}')
        elif keyword[-1] == 'community':
            keyword_pair.append(f'community:{keyword[0]}')
        elif keyword[-1] == 'year':
            keyword_pair.append(f'year:{keyword[0]}')
        elif keyword[-1] == 'month':
            keyword_pair.append(f'month:{keyword[0]}')
        elif keyword[-1] == 'enterprise':
            keyword_pair.append(f'enterprise:{keyword[0]}')

    return keyword_pair


def find_key(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]

def intent2label(intents_row):
    intents_num = {'营业利润查询': 0,
                  '企业负债查询': 1,
                  '小区成交套数查询': 2,
                  '企业营业成本查询': 3,
                  '小区绿化率查询': 4,
                  '建筑密度查询': 5,
                  '小区成交均价查询': 6,
                  '营业总收入查询': 7,
                  '地块总价查询': 8,
                  '地块成交时间查询': 9,
                  '企业债务违约查询': 10,
                  '容积率查询': 11,
                  '企业风险查询': 12,
                  '地块归属查询': 13,
                  '未知':14
                  }
    intents_label = [0] * len(intents_num)
    if '+' in intents_row:
        intents = intents_row.split('+')
        for intent in intents:
            intents_label[intents_num[intent]] = 1
    elif '+' not in intents_row:
        intent = intents_row
        intents_label[intents_num[intent]] = 1
    return intents_label



def align_tokens_with_query(tokens, query, query_slot):
    query = list(query)
    new_tokens = []
    origin_slot = query_slot
    if isinstance(origin_slot, type('str')):
        origin_slot = origin_slot.split(' ')
    new_slot = []
    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        elif token == query[0]:
            new_tokens.append(token)
            new_slot.append(origin_slot[0])
            query = query[1:]
            origin_slot = origin_slot[1:]
        elif '##' in token:
            token = token[2:]
            new_tokens.append(token)
            new_slot.append(origin_slot[0])
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
                    origin_slot = origin_slot[1:]
                    
        elif '[UNK]' == token:
            end_index = query.index(tokens[i+1])
            unk = ''.join(query[0:end_index])
            new_tokens.append(unk)
            new_slot.append(origin_slot[0])
            query = query[end_index:]
            origin_slot = origin_slot[1:]

        elif len(token) > 1:
            new_tokens.append(token)
            new_slot.append(origin_slot[0])
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
                    origin_slot = origin_slot[1:]
    return new_tokens, new_slot


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.labels = df['category']
        self.texts = [tokenizer(text, 
                                padding='max_length', 
                                max_length = 512, 
                                truncation=True,
                                return_tensors="pt") 
                      for text in df['text']]
        self.num_intents = df['intents']
        self.slots = df['slots']

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def get_batch_num_intents(self, idx):
        # Fetch a batch of inputs
        return np.array(self.num_intents[idx])

    def get_batch_slots(self, idx):
        # Fetch a batch of inputs
        return np.array(self.slots[idx])

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        batch_num = self.get_batch_num_intents(idx)
        batch_slots = self.get_batch_slots(idx)
        return batch_texts, batch_y, batch_num, batch_slots