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
import json
import pandas as pd


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(768, 15)
        self.linear2 = nn.Linear(768, 2)
        self.linear3 = nn.Linear(768, 13)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input_id, mask):
        last_hidden_state, pooled_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear1_output = self.linear1(dropout_output)
        intent_probability = self.sigmoid(linear1_output)
        num_intents = self.linear2(dropout_output)
        last_hidden_state_output = self.dropout(last_hidden_state)
        slot_probability = self.linear3(last_hidden_state_output)
        return intent_probability, num_intents, slot_probability