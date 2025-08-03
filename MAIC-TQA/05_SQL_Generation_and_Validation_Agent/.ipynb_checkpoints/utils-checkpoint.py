import json
import os
import psycopg2
from psycopg2 import sql, DatabaseError
import numpy as np
import threading
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from typing import Annotated
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from pydantic import BaseModel
from typing import Union, List, Tuple, Optional
from langchain_core.messages import ToolMessage
import re
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage
import ast

class PostgresQueryExecutor:
    def __init__(self, database, host='127.0.0.1', user='odatqa', password='odatqa123456', port="25432"):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.conn = None
        self.cur = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cur = self.conn.cursor()
        except DatabaseError as e:
            raise RuntimeError(f"Database connection error for {self.database}: {e}")

    def execute_sql(self, sql_statement):
        try:
            self.connect()
            self.cur.execute(sql_statement)
            result = self.cur.fetchall()
            self.conn.commit()
            return result
        except Exception as e:
            return f"Error executing SQL on {self.database}: {e}"
        finally:
            self.close()

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()


@tool
def sqlQueryTool(sql_statement: str, domain: str) -> str:
    """
    Execute SQL statement on the correct Postgres database based on domain.

    Domains:
    - 'realEstateSalesField' -> database='价格查询'
    - 'landInformationField' -> database='土地资产'
    - 'enterpriseFinanceField' -> database='企业财务'
    - 'unknowDomain' -> Try all databases, return the first success
    """
    domain_db_map = {
        'realEstateSalesField': '价格查询',
        'landInformationField': '土地资产',
        'enterpriseFinanceField': '企业财务'
    }

    def try_execute_on_db(database):
        executor = PostgresQueryExecutor(database=database)
        return executor.execute_sql(sql_statement)

    if domain in domain_db_map:
        db_name = domain_db_map[domain]
        result = try_execute_on_db(db_name)
        return str(result)
    elif domain == 'unknowDomain':
        for db_name in domain_db_map.values():
            result = try_execute_on_db(db_name)
            if isinstance(result, list):
                return str(result)
        return "Failed to execute SQL on all databases."
    else:
        return f"Invalid domain: {domain}"


class SQLExecutor:
    def __init__(self, host='127.0.0.1', user='odatqa', password='odatqa123456', port="25432"):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.conn = None
        self.cur = None

    def connect(self, database):
        try:
            if self.conn:
                self.conn.close()
            if self.cur:
                self.cur.close()
                
            self.conn = psycopg2.connect(
                host=self.host,
                database=database,
                user=self.user,
                password=self.password,
                port=self.port
            )
            self.cur = self.conn.cursor()
        except DatabaseError as e:
            print(f"Database connection error: {e}")
            raise

    def clean_sql(self, sql_statement):
        """
        清理SQL语句中的转义符
        """
        # Remove backslash-escaped double quotes
        cleaned_sql = sql_statement.replace('\\"', '"')
        # Remove backslash-escaped single quotes
        cleaned_sql = cleaned_sql.replace("\\'", "'")
        # Optional: Remove other common escape sequences
        cleaned_sql = cleaned_sql.replace('\\', "")
        return cleaned_sql

    def execute_sql_single_db(self, sql_statement, database):
        """
        Execute SQL on a single database
        """
        try:
            self.connect(database)
            cleaned_sql = self.clean_sql(sql_statement)
            self.cur.execute(cleaned_sql)
            result = self.cur.fetchall()
            headers = [desc[0] for desc in self.cur.description]
            self.conn.commit()
            return headers, result
        except Exception as e:
            # print(f"Error executing SQL in {database}: {e}")
            return None, None

    def execute_sql(self, sql_statement, database_hint='unKnown'):
        """
        Execute SQL statement, supporting multiple database attempts
        """
        # List of databases
        databases = ['土地资产', '企业财务', '价格查询']
        
        # If a specific database is specified, execute directly
        if database_hint in databases:
            return self.execute_sql_single_db(sql_statement, database_hint)
        
        # If database_hint is 'unKnown', try all databases in sequence
        for database in databases:
            headers, result = self.execute_sql_single_db(sql_statement, database)
            if headers is not None and result is not None:
                return headers, result
        return None, None

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()

# Store the generated table name into a JSON file
class PredictResultStorage:
    def __init__(self, file_name='testdata.json', save_lock=None):
        self.current_data = {}
        self.file_name = file_name
        self.save_lock = save_lock or threading.Lock()
    def set_data(self, data_dict):
        self.current_data.update(data_dict)
    def save_data(self):
        """将数据保存到 JSON 文件（线程安全版本）"""
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



def detect_field_category(intent):
    # Define domain classifications and their corresponding intent combinations (using frozenset for unordered matching)
    field_dict = {
        'realEstateSalesField': [
            frozenset(['小区成交套数查询']),
            frozenset(['小区成交均价查询']),
            frozenset(['小区成交均价查询', '小区成交套数查询']),
        ],
        'landInformationField': [
            frozenset(['小区绿化率查询']),
            frozenset(['建筑密度查询']),
            frozenset(['容积率查询']),
            frozenset(['地块总价查询']),
            frozenset(['地块归属查询']),
            frozenset(['地块成交时间查询']),
            frozenset(['容积率查询', '地块成交时间查询']),
            frozenset(['地块成交时间查询', '地块总价查询']),
        ],
        'enterpriseFinanceField': [
            frozenset(['企业营业成本查询']),
            frozenset(['企业风险查询']),
            frozenset(['企业负债查询']),
            frozenset(['营业总收入查询']),
            frozenset(['营业利润查询']),
            frozenset(['企业债务违约查询']),
            frozenset(['企业债务违约查询', '企业负债查询']),
            frozenset(['企业负债查询', '企业风险查询']),
        ],
    }

    # Unified format processing: Convert string to a single-element list
    if isinstance(intent, str):
        intent_set = frozenset([intent])
    elif isinstance(intent, list):
        intent_set = frozenset(intent)
    else:
        return None

    # Match the domain to which the intent belongs
    for field_name, intent_sets in field_dict.items():
        if intent_set in intent_sets:
            return field_name

    return 'unknowDomain'




def find_matched_intent(input_intent, all_intents):
    # Convert to a list of strings uniformly
    if isinstance(input_intent, str):
        input_list = [input_intent]
    elif isinstance(input_intent, list):
        input_list = input_intent
    else:
        raise ValueError("Input must be a string or a list of strings")

    input_set = set(input_list)

    for intent in all_intents:
        intent_parts = intent.split('+')
        if set(intent_parts) == input_set:
            return intent

    return None



class SQLValidationOutput(BaseModel):
    sql: str  # Actual SQL statement executed
    result: Union[List[dict], str]  # Result of SQL execution: a list of dictionaries (each row as a dict) if successful, or an error message string if failed
    note: Optional[List[Tuple[str, str, str]]] = None  # Missing or unmatched items, e.g., [("slot", "slot_type", "not found")]




def get_last_ai_message_content(response: dict) -> str:
    """
    Extract the content of the last AIMessage from response['messages'].
    """
    messages = response.get("messages", [])
    
    for msg in messages:
        if msg.__class__.__name__ == "AIMessage":
            content = msg.content
    
    return content


def get_last_ai_message_content(response: dict) -> str:
    """
    从 response["messages"] 中提取最后一个 AIMessage 的 content。
    """
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""



def agent_predict_SQL(data: dict, file_name: str, system_prompt, llm, save_lock):
    query = data['query']
    dialog = f"User:{query}\n"
    if data['dialog'] != {}:
        dialog = dialog + format_dialog(data['dialog'])
    intent = data['BERT_pred_intent-clarified']
    domain = detect_field_category(intent)
    key_word = data['BERT_pred_slots']
    table_caption = data['predict_table_caption(BM25)']
    user_prompt = f'''#########################以下为待处理的任务#####################\n<Dialog>:\n{dialog}<Domain>:{domain}\n<SLOTS>:{key_word}\n<Table_Captions>:{table_caption}'''
    agent = create_react_agent(llm, tools=[sqlQueryTool])
    try:
        response = agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        )
        ai_content = get_last_ai_message_content(response)
        structured_response = extract_by_structure(str(ai_content))
        data["predict_SQL"]  = structured_response
        data["predict_SQL_response"] = ai_content
        
    except Exception as e:
        print(f"生成SQL错误：{e}\nai_content={ai_content}hello")
        #data["predict_SQL"] = f'error:{e}'
        return
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(data)
    saver.save_data()



def follow_up_predict_SQL(data: dict, file_name: str, system_prompt, llm, save_lock):
    query = data['query']
    dialog = f"User:{query}\n"
    if data['dialog'] != {}:
        dialog = dialog + format_dialog(data['dialog'])
    intent = data['BERT_pred_intent-clarified']
    domain = detect_field_category(intent)
    key_word = data['BERT_pred_slots']
    table_caption = data['predict_table_caption(BM25)']
    user_prompt = f'''#########################以下为待处理的任务#####################\n<Dialog>:\n{dialog}<Domain>:{domain}\n<SLOTS>:{key_word}\n<Table_Captions>:{table_caption}'''
    agent = create_react_agent(llm, tools=[sqlQueryTool])
    try:
        response = agent.invoke(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
        )

        ai_content = get_last_ai_message_content(response)
        structured_response = extract_by_structure(str(ai_content))
        data["follow_up_predict_SQL"]  = structured_response
        data["follow_up_predict_SQL_response"] = ai_content
    except Exception as e:
        print(f'error:{e}')
        return
    saver = PredictResultStorage(file_name, save_lock)
    saver.set_data(data)
    saver.save_data()



def extract_by_structure(text: str) -> dict:
    result = {}

    try:
        # 找字段索引
        sql_index = text.index('"sql"')
        result_index = text.index('"result"')
        note_index = text.index('"note"')

        if not (sql_index < result_index < note_index):
            raise ValueError("The field order is incorrect, should be sql -> result -> note")

        # Extract SQL
        sql_block = text[sql_index:result_index]
        select_start = sql_block.lower().index('select')
        semicolon_end = sql_block.rindex(';')
        sql = sql_block[select_start:semicolon_end+1].strip()
        result['sql'] = sql

        # Extract result
        result_block = text[result_index:note_index]
        result_start = result_block.index(':') + 1
        result_str = result_block[result_start:].strip().rstrip(',').strip()

        try:
            parsed_result = json.loads(result_str)
            if parsed_result == '[]':
                parsed_result = []
            result['result'] = parsed_result
        except Exception:
            result['result'] = result_str  #Keep the original string

        # Extract note
        note_block = text[note_index:]
        note_start = note_block.index('[')
        note_end = note_block.rindex(']')
        note_str = note_block[note_start:note_end+1]
        result['note'] = json.loads(note_str)

        return result

    except Exception as e:
        raise ValueError(f"Parsing failed: {e}")

def format_dialog(dialog_dict):
    """
    Format a dialogue dictionary with structure like {"Q-0": ..., "A-0": ..., "Q-1": ..., ...}
    into a string in the format:
        System: ...
        User: ...
    and sort it in ascending order by number.
    """
    from collections import defaultdict

    # Group Q/A by number
    dialog_by_num = defaultdict(dict)

    for k, v in dialog_dict.items():
        if '-' in k:
            prefix, num = k.split('-')
            if num.isdigit():
                dialog_by_num[int(num)][prefix] = v.strip()

    # Sort by number and output in Q-A format
    formatted_dialog = []
    for num in sorted(dialog_by_num.keys()):
        pair = dialog_by_num[num]
        if 'Q' in pair:
            formatted_dialog.append(f"System:{pair['Q']}")
        if 'A' in pair:
            formatted_dialog.append(f"User:{pair['A']}")

    return "\n".join(formatted_dialog)



def extract_sql_content(input_string):
    # Try with both possible prefixes
    prefixes = ['<SQL>:', ' <SQL>:']
    for prefix in prefixes:
        start_index = input_string.find(prefix)
        if start_index != -1:
            # Add the length of the prefix to get to the start of the SQL content
            start_index += len(prefix)
            sql_content = input_string[start_index:]
            return sql_content



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