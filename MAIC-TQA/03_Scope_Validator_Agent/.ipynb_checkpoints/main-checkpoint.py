import numpy as np
import pandas as pd
from ipywidgets import FloatProgress
import re
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
import json
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
import os
from tqdm import tqdm
import ast
import re
from rank_bm25 import BM25Okapi
from tabulate import tabulate
import jieba
import psycopg2
from psycopg2 import sql, DatabaseError
from utils import *
from clarification_interface import *
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict, List
from tabulate import tabulate

def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='SLU')
    parser.add_argument('--task', type=str, default='test', help='transform, eval or prediction')
    parser.add_argument('--model_name', type=str, default='qwen3-32b', help='LLM model name')
    parser.add_argument('--SLU_model_name', type=str, default='BERT', help='SLU task model name')
    parser.add_argument('--base_url', type=str, default='http://your_api_url', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='your_api_key', help='LLM API KEY')
    parser.add_argument('--workers', type=int, default=15, help='Request for the number of concurrent LLMs.')
    return parser

def slotSearchTool(slots, domain):
    """
    Search for whether slots appear in the table headers of a specified domain.
 
    Args:
        slots (str or List[str]): The string(s) to search for.
        domain (str): The domain of the question, one of 'realEstateSalesField', 
                      'landInformationField', or 'enterpriseFinanceField'.
 
    Returns:
        Dict[str, int]: A dictionary where keys are the input slot strings and values
                        are the number of matching tables found (0, 1, or n).
    """
    field_to_caption_key = {
        'realEstateSalesField': 'community_table_captions',
        'landInformationField': 'land_table_captions',
        'enterpriseFinanceField': 'enterprise_table_captions'
    }
    if domain not in field_to_caption_key:
        raise ValueError(f"Unknown domain: {domain}")

    if isinstance(slots, str):
        slots = [slots]

    captions = all_table_captions[field_to_caption_key[domain]]
    result = {}
    for slot in slots:
        count = sum(1 for caption in captions if slot in caption)
        result[slot] = count
    #print(result)
    return result


# 使用方法
parser = create_parser()
args = parser.parse_args()

BASE_URL = args.base_url
MODEL_NAME = args.model_name
OPENAI_API_KEY = args.api_key
CONCURRENT_WORKERS = args.workers
SLU_MODEL_NAME = args.SLU_model_name

global_save_lock = threading.Lock()
all_table_captions = {}
def main():
    global all_table_captions
    if args.task == 'prediction':
        print(f'Connecting to LLM-{args.model_name}...')
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            base_url=BASE_URL
        )

        template_json_file = '../prompts/prompts.json'
        with open(template_json_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        table_caption_file_path = '../../datasets/table_captions/all_table_captions.json'
        with open(table_caption_file_path, 'r', encoding='utf-8') as f:
            all_table_captions = json.load(f)

        result_saved_file = f'./results/{MODEL_NAME} Slots Check Results.json'
        json_file = f'../02_SLU_module/results/test_intent_slot_pred_bert_clarified.json'
        with open(json_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        # Retrieve the set of processed queries
        processed_queries = set()
        if os.path.exists(result_saved_file):
            try:
                with open(result_saved_file, 'r') as f:
                    existing_data = json.load(f)
                    processed_queries = {item['query'] for item in existing_data if 'query' in item}
            except json.JSONDecodeError:
                pass
        # Filter out unprocessed data
        remaining_datas = [data for data in datas if data.get('query') not in processed_queries]
        # Multithreaded processing function
        def process_data(data):
            try:
                system_prompt = templates['Slots_Check_System_Prompt']
                agent = create_react_agent(
                    model=llm,
                    tools=[slotSearchTool],
                )
                check_slots(data=data, all_table_captions=all_table_captions,file_name=result_saved_file, agent=agent, system_prompt=system_prompt, save_lock = global_save_lock)
                return True
            except Exception as e:
                print(f"Error processing query '{data.get('query', 'unknown')}': {str(e)}")
                return False
        # rocess using a thread pool in parallel
        with tqdm(total=len(remaining_datas), desc="处理进度", unit="query") as pbar:
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [executor.submit(process_data, data) for data in remaining_datas]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Handle exceptions: {str(e)}")
                        continue
                    pbar.update(1)
        print("All data processing completed!")

    elif args.task == 'follow-up-generation':
        result_saved_file = f'./results/{MODEL_NAME} Slots Check Results.json'
        with open(result_saved_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        for data in tqdm(datas):
            true_triplets = [tuple(item) for item in data['scope_ambiguity']]
            pred_triplets = [tuple(item) for item in data['slot_check_result']]
            true_set = set(true_triplets)
            pred_set = set(pred_triplets)
            matched = true_set & pred_set
            if 'dialog' not in data.keys():
                data['dialog'] = {}
            for item in matched:
                if item[2] != 'Correct' and item[1] == 'district':
                    follow_answer = FROM_AmbiguityClarification_interface(item, data)
                    if follow_answer != '无':
                        if item[0] == '':
                            data['dialog']['Q-2'] = f"你的输入缺少区域名称，请提供一个正确的区域。"
                        elif item[0] != '':
                            data['dialog']['Q-2'] = f"你输入的区域名称{item[0]}不正确，请提供一个正确的区域。"
                        data['dialog']['A-2'] = follow_answer
                if item[2] != 'Correct' and item[1] == 'city':
                    follow_answer = FROM_AmbiguityClarification_interface(item, data)
                    if follow_answer != '无':
                        if item[0] == '':
                            data['dialog']['Q-1'] = f"你的输入缺少城市名称，请提供一个正确的城市。"
                        elif item[0] != '':
                            data['dialog']['Q-1'] = f"你输入的城市名称{item[0]}不正确，请提供一个正确的城市。"
                        data['dialog']['A-1'] = follow_answer
                if item[2] != 'Correct' and item[1] == 'year':
                    follow_answer = FROM_AmbiguityClarification_interface(item, data)
                    if follow_answer != '无':
                        if item[0] == '':
                            data['dialog']['Q-3'] = f"你的输入中没有包含年份信息，请提供一个为在2019-2022年之间的年份。"
                        elif item[0] != '':
                            data['dialog']['Q-3'] = f"你输入的{item[0]}不存在，请提供一个为在2019-2022年之间的年份。"
                        data['dialog']['A-3'] = follow_answer
        with open(result_saved_file, 'w', encoding='utf-8') as f:
            json.dump(datas, f, indent=4, ensure_ascii=False)

    elif args.task == 'slot-check-eval':
        result_saved_file = f'./results/{MODEL_NAME} Slots Check Results.json'
        with open(result_saved_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        accuracy_count = 0
        total_count = 0
        TP = 0
        FP = 0
        FN = 0
        for data in datas:
            # Filter out elements from slot_check_result that meet the specified conditions.
            pred = [x for x in data["slot_check_result"] if len(x) > 2 and x[2] != "Correct"]
            # Filter out elements from scope_ambiguity that meet the specified conditions.
            true = [x for x in data["scope_ambiguity"] if len(x) > 2 and x[2] != "Correct"]
            # Accuracy calculation
            if pred == true:
                accuracy_count += 1
            total_count += 1
            # P/R/F1 calculation
            if true:  # Only count samples where scope_ambiguity is non-empty.
                pred_set = set(tuple(x) for x in pred)
                true_set = set(tuple(x) for x in true)
                TP += len(pred_set & true_set)
                FP += len(pred_set - true_set)
                FN += len(true_set - pred_set)
        # Accuracy is calculated over all samples.
        acc = accuracy_count / total_count if total_count else 0
        # P/R/F1 are calculated only on samples with annotated triplets (excluding the Correct label).
        P = TP / (TP + FP) if (TP + FP) else 0
        R = TP / (TP + FN) if (TP + FN) else 0
        F1 = 2 * P * R / (P + R) if (P + R) else 0
        table = [
            ["Accuracy",  f"{acc:.4f}"],
            ["Precision",  f"{P:.4f}"],
            ["Recall", f"{R:.4f}"],
            ["F1 Score",  f"{F1:.4f}"]
        ]
        print('#########################################\n')
        print('Slot check results (excluding "Correct")\n')
        print(tabulate(table, tablefmt="grid"))
        print('\n#########################################')

if __name__ == "__main__":
    main()