import numpy as np
import pandas as pd
from ipywidgets import FloatProgress
import re
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
import os
import ast
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


def create_parser():
    # basic config
    parser = argparse.ArgumentParser(description='SLU')
    parser.add_argument('--task', type=str, default='test', help='transform, eval or prediction')
    parser.add_argument('--model_name', type=str, default='qwen3-32b', help='LLM model name')
    parser.add_argument('--base_url', type=str, default='http://your_api_url', help='LLM base url')
    parser.add_argument('--api_key', type=str, default='your_api_key', help='LLM API KEY')
    parser.add_argument('--workers', type=int, default=15, help='Request for the number of concurrent LLMs.')
    return parser

# 使用方法
parser = create_parser()
args = parser.parse_args()

BASE_URL = args.base_url
MODEL_NAME = args.model_name
OPENAI_API_KEY = args.api_key
CONCURRENT_WORKERS = args.workers

global_save_lock = threading.Lock()
def main():
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

        result_saved_file = f'./results/{MODEL_NAME} SQL Prediction Results.json'
        json_file = f'../04_Table_Retrieval_Agent/results/{MODEL_NAME} Table Caption Prediction Results.json'
        with open(json_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        # Get the set of already processed queries
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
                all_intents = templates['SQL_Generation_Prompt'].keys()
                intent = find_matched_intent(data['BERT_pred_intent-clarified'], all_intents)
                system_prompt = templates['SQL_Generation_Prompt'][intent]
                agent_predict_SQL(data=data, file_name=result_saved_file, system_prompt=system_prompt, llm=llm, save_lock=global_save_lock)
                return True
            except Exception as e:
                print(f"Error processing query '{data.get('query', 'unknown')}': {e}")
                return False
        # Use a thread pool to process tasks in parallel
        with tqdm(total=len(remaining_datas), desc="Processing progress", unit="query") as pbar:
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [executor.submit(process_data, data) for data in remaining_datas]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Processing exception: {str(e)}")
                        continue
                    pbar.update(1)
        print("All data processing completed!")

    elif args.task == 'follow_prediction':
        result_saved_file = f'./results/{MODEL_NAME} SQL Prediction Results(Follow-up).json'
        json_file = f'./results/{MODEL_NAME} SQL Prediction Results.json'
        if os.path.exists(result_saved_file):
            with open(result_saved_file, 'r', encoding='utf-8') as f:
                try:
                    saved_data = json.load(f)
                except json.JSONDecodeError:
                    saved_data = []
        else:
            saved_data = []

        # Create a set for quick exclusion of already processed queries
        processed_queries = set(item.get('query') for item in saved_data if 'query' in item)

        with open(json_file, 'r', encoding='utf-8') as f:
            datas = json.load(f)
        follow_datas = []
        intersections = []
        for data in datas:
            if (
                'predict_SQL' in data and isinstance(data['predict_SQL'], dict)
                and 'note' in data['predict_SQL']
                and 'condition_ambiguity' in data
            ):
                note = data['predict_SQL']['note']
                labels = data['condition_ambiguity']
        
                if note is None:
                    note = []
                note_set = set(tuple(item) for item in note)
                labels_set = set(tuple(item) for item in labels)
                overlap = note_set & labels_set
                if overlap and data.get('query') not in processed_queries:
                    for item in overlap:
                        follow_answer = WHERE_AmbiguityClarification_interface(item, data)
                        if follow_answer != '无':
                            data['dialog']['Q-5'] = f"你输入的{item[0]}：{item[1]}未查询到，请补充一个正确的{item[0]}。"
                            data['dialog']['A-5'] = follow_answer
                            follow_datas.append(data)
                            break  # Avoid adding the same data multiple times (e.g., when overlap has multiple items)
        print(f'Number of follow-up questions needed: {len(follow_datas)}')
        # Multithreading processing function
        print(f'Connecting to LLM-{args.model_name}...')
        llm = ChatOpenAI(
            model_name=MODEL_NAME,
            openai_api_key=OPENAI_API_KEY,
            base_url=BASE_URL
        )

        template_json_file = '../prompts/prompts.json'
        with open(template_json_file, 'r', encoding='utf-8') as f:
            templates = json.load(f)
        # Multithreaded processing function
        def process_data(data):
            try:
                all_intents = templates['SQL_Generation_Prompt'].keys()
                intent = find_matched_intent(data['BERT_pred_intent-clarified'], all_intents)
                system_prompt = templates['SQL_Generation_Prompt'][intent]
                follow_up_predict_SQL(data=data, file_name=result_saved_file, system_prompt=system_prompt, llm=llm, save_lock=global_save_lock)
                return True
            except Exception as e:
                print(f"Error processing query '{data.get('query', 'unknown')}': {str(e)}")
                return False
        # Use thread pool for parallel processing
        with tqdm(total=len(follow_datas), desc="处理进度", unit="query") as pbar:
            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [executor.submit(process_data, data) for data in follow_datas]
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Handling exception: {str(e)}")
                        continue
                    pbar.update(1)
        print("All data processing completed!")

    elif args.task == 'eval':
        result_saved_file = f'./results/{MODEL_NAME} SQL Prediction Results.json'
        with open(result_saved_file, 'r', encoding='utf-8') as f:
            origin_datas = json.load(f)
        follow_up_result_saved_file = f'./results/{MODEL_NAME} SQL Prediction Results(Follow-up).json'
        with open(follow_up_result_saved_file, 'r', encoding='utf-8') as f:
            follow_datas = json.load(f)

        # Use query as the unique identifier field
        key_field = 'query'
        # Build a mapping dictionary for follow_datas, prioritizing content from follow_datas
        merged_dict = {data[key_field]: data for data in origin_datas}
        merged_dict.update({data[key_field]: data for data in follow_datas})
        # Convert to a list
        datas = list(merged_dict.values())

        TP = 0
        total_pred = 0
        total_true = 0
        acc_count = 0
        total = len(datas)
        for data in datas:
            pred = data.get('predict_SQL', {}).get('note', []) or []
            true = data.get('condition_ambiguity', [])
            # Accuracy: Perfectly match
            if set(tuple(p) for p in pred) == set(tuple(t) for t in true):
                acc_count += 1
            # Triple-level matching
            pred_set = set(tuple(p) for p in pred)
            true_set = set(tuple(t) for t in true)
            TP += len(pred_set & true_set)
            total_pred += len(pred_set)
            total_true += len(true_set)
        acc = acc_count / total if total else 0.0
        P = TP / total_pred if total_pred else 0.0
        R = TP / total_true if total_true else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        table = [
            ["Accuracy",  f"{acc:.4f}"],
            ["Precision",  f"{P:.4f}"],
            ["Recall", f"{R:.4f}"],
            ["F1 Score",  f"{F1:.4f}"]
        ]
        print('#########################################\n')
        print('WHERE clause check results\n')
        print(tabulate(table, tablefmt="grid"))
        print('\n#########################################')

        total_query = 0
        sqlQueryExtractable = 0

        for data in tqdm(datas):
            total_query += 1
            predict_SQL = ''
        
            if "follow_up_predict_SQL" in data:
                try:
                    predict_SQL = data["follow_up_predict_SQL"].get('sql', '')
                    if predict_SQL:
                        sqlQueryExtractable += 1
                except Exception:
                    pass
        
            elif "predict_SQL" in data:
                try:
                    predict_SQL = data["predict_SQL"].get('sql', '')
                    if predict_SQL:
                        sqlQueryExtractable += 1
                except Exception:
                    pass

            # Determine the database type
            if '土地成交信息表' in predict_SQL:
                database = '土地资产'
            elif '全国企业' in predict_SQL and '年财务表' in predict_SQL:
                database = '企业财务'
            elif '商品房成交价格表' in predict_SQL:
                database = '价格查询'
            else:
                database = 'unKnown'
            if predict_SQL != '':
                try:
                    executor = SQLExecutor()
                    headers, result = executor.execute_sql(predict_SQL, database)

                    if result != None:
                        data['llm_predict_sql_result'] = result
                    else:
                        data['llm_predict_sql_result'] = 'error'
                except Exception as e:
                    data['llm_predict_sql_result'] = 'error'
                finally:
                    data['llm_predict_sql'] = predict_SQL
                    executor.close()
            else:
                data['llm_predict_sql_result'] = 'error'
                data['llm_predict_sql'] = predict_SQL
        print(f'Total QA count: {total_query}, Number of extractable SQL: {sqlQueryExtractable}')

        preds = []
        trues = []
        unexecutable_sql = 0
        all_sql = 0
        for result in tqdm(datas):
            if result['llm_predict_sql_result'] == 'error' or result['llm_predict_sql_result'] == None:
                result['llm_predict_sql_result'] = []
                unexecutable_sql += 1
            preds.append(result['llm_predict_sql'])
            trues.append(result['SQL'])
            all_sql += 1
        ECR = (all_sql - unexecutable_sql)/all_sql
        print(f"ECR:{ECR}")
        uuids = {}
        for result in tqdm(datas):
            table_results = result['llm_predict_sql_result']
            true_SQL_answer = result['SQL_answer']
            if true_SQL_answer == None:
                true_SQL_answer = []
            if table_results is not None and all(isinstance(i, list) for i in table_results):
                table_results = [tuple(sublist) for sublist in table_results]
            if true_SQL_answer is not None and all(isinstance(i, list) for i in true_SQL_answer):
                true_SQL_answer = [tuple(sublist) for sublist in true_SQL_answer]
            if table_results != None and set(table_results) == set(true_SQL_answer) and len(table_results) == len(true_SQL_answer):
                result['predict_correctness'] = True
            else:
                result['predict_correctness'] = False
            if result['uuid'] not in uuids:
                uuids[result['uuid']] = []
            uuids[result['uuid']].append(result['predict_correctness'])

        for key in uuids.keys():
            c = sum(uuids[key])
            n = len(uuids[key])
            uuids[key] = {'c': c, 'n': n}
            uuids[key]['pass1'] = pass_at_k(n = uuids[key]['n'], c = uuids[key]['c'],k=1)

        print(f'pass@1：{calculate_average_pass1(uuids)}')

if __name__ == "__main__":
    main()