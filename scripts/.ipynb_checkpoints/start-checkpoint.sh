#!/bin/bash

# Set default model name and number of worker threads
MODEL_NAME="qwen3-32b"
WORKERS=10  # Default number of worker threads

# 检查传入参数
if [ $# -ge 1 ]; then
    MODEL_NAME=$1
fi
if [ $# -ge 2 ]; then
    WORKERS=$2
fi


echo "Running evaluation for model: ${MODEL_NAME} with workers: ${WORKERS}"
echo "Start time: $(date)"

# Declare baseURL and APIkey variables
baseURL="https://api.moonshot.cn/v1"  # Replace with the actual base URL
APIkey="replace-with-your-API-key"          # Replace with the actual API key



# BERT-based Intent and Slot Prediction
echo "=== Running Intent and Slot Prediction tasks ==="
cd ../MAIC-TQA/01_BERT_finetune
python main.py --task='test'


# Intent Disambiguation
echo "=== Running SELECT Disambiguation tasks ==="
cd ../02_SLU_module
python main.py



# Slot Validation Command
echo "=== Running FROM Disambiguation tasks ==="
cd ../03_Scope_Validator_Agent
python main.py --task='prediction' --model_name="${MODEL_NAME}" --base_url="${baseURL}" --api_key="${APIkey}"
python main.py --task='follow-up-generation' --model_name="${MODEL_NAME}"
python main.py --task='slot-check-eval' --model_name="${MODEL_NAME}"


# Retrieve Table Command
echo "=== Running Table Retrivel tasks ==="
cd ../04_Table_Retrieval_Agent
python main.py --task='prediction' --model_name="${MODEL_NAME}" --workers="${WORKERS}" --base_url="${baseURL}" --api_key="${APIkey}"
python main.py --task='eval' --model_name="${MODEL_NAME}"


# Final_Prediction Directory Command
echo "=== Running WHERE Disambiguation tasks ==="
cd ../05_SQL Generation_and_Validation_Agent
python main.py --task='prediction' --model_name="${MODEL_NAME}" --workers="${WORKERS}" --base_url="${baseURL}" --api_key="${APIkey}"
python main.py --task='follow_prediction' --model_name="${MODEL_NAME}" --workers="${WORKERS}" --base_url="${baseURL}" --api_key="${APIkey}"
python main.py --task='eval' --model_name="${MODEL_NAME}"