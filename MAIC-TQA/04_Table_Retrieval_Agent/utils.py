import json
import os
import threading
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List
import json
import re

# 将生成的表格名称存储到json文件当中
class PredictResultStorage:
    def __init__(self, file_name='testdata.json', save_lock=None):
        self.current_data = {}
        self.file_name = file_name
        self.save_lock = save_lock or threading.Lock()
    def set_data(self, data_dict):
        self.current_data.update(data_dict)
    def save_data(self):
        with self.save_lock:
            if os.path.exists(self.file_name):
                with open(self.file_name, 'r+') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    data.append(self.current_data)
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            else:
                with open(self.file_name, 'w') as f:
                    json.dump([self.current_data], f, indent=4)
            self.current_data = {}

def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]


def extract_content_after_summary(response):
    keyword = '<Summary>:'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]

def predict_table_caption(data: dict, file_name: str, chain, save_lock):
    query = data['query']
    try:
        dialog = f"User:{query}\n"
        if data['dialog'] != {}:
            dialog = dialog + format_dialog(data['dialog'])
        response = chain.invoke({"query": dialog})
        response = response.content
        if '</think>' in response:
            response = extract_content_after_think(response)
        summary = extract_summary_from_response(response)

        if summary == None:
            print(f"No result extracted:{response}")
            return
    except Exception as e:
        print(f"API call failed, skipped save operation\ndata:\n{data}\n\nERROR：{str(e)}")
        return
    predicted_table_name = summary
    # Save data (this is only executed if the API call succeeds)
    data["predict_table_caption"] = predicted_table_name
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(data)
    saver.save_data()


def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]


def extract_summary_from_response(response: str):
    """
    Extract summary from response string, supports multiple formats
    """
    # Handle cases with <Summary> tags
    summary_tag_match = re.search(r'<Summary>:\s*($$.*?$$)', response)
    if summary_tag_match:
        try:
            summary = json.loads(summary_tag_match.group(1))
            if isinstance(summary, list) and len(summary) > 0:
                return summary[0]
        except json.JSONDecodeError:
            pass

    # Handle direct list format
    list_match = re.search(r'$$.*?"(.*?)".*?$$', response)
    if list_match:
        return list_match.group(1)

    # Handle JSON format
    json_matches = re.findall(r"\{[\s\S]*?\}", response)
    for match in json_matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and "Summary" in parsed:
                return parsed["Summary"]
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]
        except json.JSONDecodeError:
            continue

    # If all the above methods fail, attempt to extract the last string that looks like a title
    title_match = re.search(r'"([^"]+年[^"]+表)"', response)
    if title_match:
        return title_match.group(1)

    return None



class SummarySchema(BaseModel):
    Summary: List[str]



def format_dialog(dialog_dict):
    """
    Convert a dialogue dictionary in the format like {"Q-0": ..., "A-0": ..., "Q-1": ..., ...}
    into a string in the format:
    System: ...
    User: ...
    in chronological order based on the numbering.
    """
    from collections import defaultdict

    dialog_by_num = defaultdict(dict)

    for k, v in dialog_dict.items():
        if '-' in k:
            prefix, num = k.split('-')
            if num.isdigit():
                dialog_by_num[int(num)][prefix] = v.strip()
    formatted_dialog = []
    for num in sorted(dialog_by_num.keys()):
        pair = dialog_by_num[num]
        if 'Q' in pair:
            formatted_dialog.append(f"System:{pair['Q']}")
        if 'A' in pair:
            formatted_dialog.append(f"User:{pair['A']}")

    return "\n".join(formatted_dialog)





def align_tokens_with_query(tokens, query):
    query = list(query)
    new_tokens = []
    for i, token in enumerate(tokens):
        if token == '[CLS]' or token == '[SEP]':
            continue
        elif token == query[0]:
            new_tokens.append(token)
            query = query[1:]
        elif '##' in token:
            token = token[2:]
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
                    
        elif '[UNK]' == token:
            end_index = query.index(tokens[i+1])
            unk = ''.join(query[0:end_index])
            new_tokens.append(unk)
            query = query[end_index:]

        elif len(token)>1:
            new_tokens.append(token)
            for t in list(token):
                if t == query[0]:
                    query = query[1:]
    return new_tokens    

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

    return keywords

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


def metric_compute(trues: list, preds: list):
    if len(trues) != len(preds):
        return 'Input lengthes not equal!'
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    for true_label, pred_label in zip(trues, preds):
        if isinstance(true_label, type('')):
            true_label = [true_label]
        if isinstance(pred_label, type('')):
            pred_label = [pred_label]
        for pred in pred_label:
            if pred in true_label:
                precision += 1
            precision_all += 1
        for true in true_label:
            if true in pred_label:
                recall += 1
            recall_all += 1
    P = precision/precision_all
    R = recall/recall_all
    F1 = 2 * P * R / (P + R)
    print("Evaluation Metrics:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"│ Precision (P) │ {P:.4f} │")
    print(f"│ Recall (R)    │ {R:.4f} │")
    print(f"│ F1 Score      │ {F1:.4f} │")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return P, R, F1
