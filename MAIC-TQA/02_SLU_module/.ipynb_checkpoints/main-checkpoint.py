import re
import json
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import pandas as pd
from model import *
from clarification_interface import *


def find_key(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]

file_path = '../01_BERT_finetune/results/test_intent_slot_pred_bert.json'
with open(file_path, 'r', encoding='utf-8') as f:
    datas = json.load(f)



model_path = '../01_BERT_finetune/models_pt/bert.pt'
model = torch.load(model_path)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    model = model.cuda()
BERT_PATH = '/nas/LLMs/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

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
all_intents = [ '营业利润查询',
                '企业负债查询',
                '小区成交套数查询',
                '企业营业成本查询',
                '小区绿化率查询',
                '建筑密度查询',
                '小区成交均价查询',
                '营业总收入查询',
                '地块总价查询',
                '地块成交时间查询',
                '企业债务违约查询',
                '容积率查询',
                '企业风险查询',
                '地块归属查询',
                '未知'
              ]

for data in datas:
    if data['BERT_pred_intent'] == '未知' or data['BERT_pred_intent'] == ['未知']:
        user_clarification = SELECT_AmbiguityClarification_interface(data['query'], data)
        query = f'''User:{data['query']}User:{user_clarification}'''
        encoded_text = tokenizer(query, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
        encoded_text_id = encoded_text['input_ids'].to(device)
        mask = encoded_text['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model(encoded_text_id, mask)
        intent_probility = outputs[0].view(-1)
        _, intent_idx = torch.topk(intent_probility, k=2, dim=0)
        intent_idx = intent_idx.cpu()
        intent_num = outputs[1].argmax().cpu()
        if intent_num == 0:
            intent_idx = intent_idx[0]
            intent = find_key(intents_num, intent_idx)[0]
        elif intent_num == 1:
            intent = [find_key(intents_num, i)[0] for i in intent_idx]
            if intent[0]+'+'+intent[1] in all_intents:
                intent = intent[0]+'+'+intent[1]
            if intent[1]+'+'+intent[0] in all_intents:
                intent = intent[1]+'+'+intent[0]
        data["BERT_pred_intent-clarified"] = intent
        if 'dialog' not in data.keys():
            data['dialog'] = {}
        if user_clarification != '无':
            data['dialog']['Q-0'] = '你的问题不够详细，需要进一步补充信息，才能确定。'
            data['dialog']['A-0'] = user_clarification
    else:
        data["BERT_pred_intent-clarified"] = data["BERT_pred_intent"]

save_json_file = './results/test_intent_slot_pred_bert_clarified.json'
with open(save_json_file, 'w', encoding='utf-8') as file:
    json.dump(datas, file, ensure_ascii=False, indent=4)