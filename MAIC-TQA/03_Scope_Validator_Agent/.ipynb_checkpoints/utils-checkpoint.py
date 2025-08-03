import json
import os
import numpy as np
import threading
import re
from langchain_core.messages import AIMessage
import ast

# Store the generated table name into a JSON file.
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


def from_slots_to_caption(field, slots, table_caption_needed_slot, all_table_captions):
    # Mapping from domain to table title key
    field2caption_key = {
        'landInformationField': 'land_table_captions',
        'realEstateSalesField': 'community_table_captions',
        'enterpriseFinanceField': 'enterprise_table_captions'
    }
    # Retrieve the required slot types for this domain.
    needed_slot_types = set(table_caption_needed_slot.get(field, []))
    # Retrieve all table captions for this domain.
    captions = all_table_captions.get(field2caption_key.get(field, ''), [])

    result = {}
    for slot in slots:
        if ':' not in slot:
            continue
        slot_type, slot_value = slot.split(':', 1)
        if slot_type not in needed_slot_types:
            continue  # Skip irrelevant slot types.
        # Retrieve table titles (fuzzy inclusion).
        matched_captions = [caption for caption in captions if slot_value in caption]
        result[slot_value] = matched_captions
    # If all results are empty, return an empty dictionary.
    if not result or all(len(v) == 0 for v in result.values()):
        return {}
    return result




def check_slots(data: dict, all_table_captions: dict, file_name: str, agent, system_prompt, save_lock):
    query = data['query']
    realEstateSalesField  = ['小区成交套数查询', '小区成交均价查询']
    landInformationField  = ['小区绿化率查询', '建筑密度查询', '容积率查询', '地块总价查询', '地块归属查询', '地块成交时间查询']
    enterpriseFinanceField  = ['企业营业成本查询', '企业风险查询', '企业负债查询', '营业总收入查询', '营业利润查询', '企业债务违约查询']
    pred_intent = data['BERT_pred_intent-clarified']
    pred_slots = data['BERT_pred_slots']
    table_caption_needed_slot = {'realEstateSalesField': ['city', 'district'], 'landInformationField': ['city', 'district'],'enterpriseFinanceField': ['year']}
    if pred_intent[0] in realEstateSalesField:
        domain ='realEstateSalesField'
        needed_slot_types = table_caption_needed_slot[domain]
    elif pred_intent[0] in landInformationField:
        domain ='landInformationField'
        needed_slot_types = table_caption_needed_slot[domain]
    elif pred_intent[0] in enterpriseFinanceField:
        domain ='enterpriseFinanceField'
        needed_slot_types = table_caption_needed_slot[domain]
    else:
        domain ='unknowDomain'
        needed_slot_types = ['city', 'district', 'year']
    try:
        user_input = str(f'<问题>：{query}\n<问题领域>：{domain}\n<槽位信息>：{pred_slots}\n<目标槽位类型>：{needed_slot_types}')
        #messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}, {"role": "user", "content": '/no_think'}]
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
        response = agent.invoke({ "messages": messages})
        # 输出中的结构化结果
        result = get_last_ai_message_content(response)
        format_answer  = extract_list_of_list(result)

        last_message = response['messages'][-1]
        if isinstance(last_message, AIMessage):
            response_answer = last_message.content
        else:
            response_answer = last_message.get("content", "")
        
        response_answer = extract_classification_from_text(response_answer)
        if format_answer:
            data["slot_check_result"] = format_answer
        elif response_answer:
            data["slot_check_result"] = response_answer
        else:
            print(f'No answer was extracted.：\nresponse_answer:{response_answer},\n:format_answer{format_answer}')
            return

    except Exception as e:
        print(f"API call failed, save operation skipped | Error details: {str(e)}")
        return
    
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(data)
    saver.save_data()


def get_last_ai_message_content(response: dict) -> str:
    """
    Extract the content from the last AIMessage in response['messages'].
    """
    messages = response.get("messages", [])
    
    for msg in messages:
        if msg.__class__.__name__ == "AIMessage":
            content = msg.content
    
    return content



def extract_list_of_list(content):
    if isinstance(content, list):
        return content
    if isinstance(content, str):
        try:
            result = ast.literal_eval(content)
            if isinstance(result, list):
                return result
        except Exception as e:
            return []
    return []



def extract_classification_from_text(text: str):
    """
    Extract classification results in the format [['slot', 'type', 'label'], ...] from the LLM's normal output text.
    """
    # 支持中括号嵌套格式的匹配
    pattern = r"\[\s*\[.*?\]\s*\]"  # Match text in the format [[...], [...]]
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            # Safely convert to a list using literal_eval
            from ast import literal_eval
            result = literal_eval(match)
            if all(isinstance(item, list) and len(item) == 3 for item in result):
                return result
        except Exception:
            continue
    return []



def extract_result_content(input_string):
    # Try with both possible prefixes
    prefixes = ['<分类结果>:', ' <分类结果>:']
    
    for prefix in prefixes:
        start_index = input_string.find(prefix)
        if start_index != -1:
            # Add the length of the prefix to get to the start of the SQL content
            start_index += len(prefix)
            sql_content = input_string[start_index:]
            return sql_content


def metric_compute(trues: list, preds: list):
    if len(trues) != len(preds):
        return 'Input lengthes not equal!'
    precision = 0
    precision_all = 0
    recall = 0
    recall_all = 0
    accuracy = 0
    acc_all = 0
    for true_label, pred_label in zip(trues, preds):
        if isinstance(true_label, type('')):
            true_label = [true_label]
        if isinstance(pred_label, type('')):
            pred_label = [pred_label]
        if true_label == pred_label:
            accuracy += 1
        acc_all += 1
        for pred in pred_label:
            if pred in true_label:
                precision += 1
            precision_all += 1
        # 查全率
        for true in true_label:
            if true in pred_label:
                recall += 1
            recall_all += 1
    acc = accuracy/acc_all
    P = precision/precision_all
    R = recall/recall_all
    F1 = 2 * P * R / (P + R)
    return acc, P, R, F1



def extract_content_after_think(response):
    keyword = '</think>'
    start_index = response.find(keyword)
    if start_index == -1:
        return ''
    return response[start_index + len(keyword):]



def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k /np.arange(n - c + 1, n + 1))

def calculate_average_pass1(data):
    total_pass1 = sum(item['pass1'] for item in data.values())
    count = len(data)
    if count == 0:
        return 0 
    return total_pass1 / count