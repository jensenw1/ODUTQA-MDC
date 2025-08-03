import argparse
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from utils import *
from model import *



def parse_args():
    parser = argparse.ArgumentParser(description='BERT classifier training and evaluation')
    # Path-related arguments
    parser.add_argument('--bert_path', type=str, default='your_path_to/bert-base-chinese', help='Path to the pre-trained BERT model')
    parser.add_argument('--root_path', type=str, default='../../datasets/', help='Root directory of dataset JSON files')
    parser.add_argument('--model_save_path', type=str, default='./models_pt/bert.pt', help='Path to save the trained model')
    parser.add_argument('--log_dir', type=str, default='./runs', help='Directory for TensorBoard log files')
    parser.add_argument('--task', type=str, default='train', help='Task type')
    # Training-related arguments
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    return parser.parse_args()

def main():
    args = parse_args()
    # Define mappings for slots and intents
    slots_num = {'O': 0,
          'B-year': 1,
          'I-year': 2,
          'B-month': 3,
          'I-month': 4,
          'B-city': 5,
          'I-city': 6,
          'B-district': 7,
          'I-district': 8,
          'B-community': 9,
          'I-community': 10,
          'B-enterprise': 11,
          'I-enterprise': 12
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
    # Initialize the tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.task == 'train':
        # Load the dataset
        train_path = args.root_path + 'train.json'
        with open(train_path, 'r', encoding='utf-8') as f:
            all_datas = json.load(f)
        df_train = json2dataframe(all_datas, tokenizer, slots_num, intents_num)
        validation_path = args.root_path + 'validation.json'
        with open(validation_path, 'r', encoding='utf-8') as f:
            all_datas = json.load(f)
        df_val = json2dataframe(all_datas, tokenizer, slots_num, intents_num)
        print(f"Train Num:{len(df_train)}, Validation Num:{len(df_val)}")
        # Initialize TensorBoard and model
        writer = SummaryWriter(args.log_dir)
        model = BertClassifier()
        # Train and evaluate
        train(model, df_train, df_val, args.lr, args.epochs, writer, tokenizer, args.batch_size)
        # Save the model
        torch.save(model, args.model_save_path)
        print(f"The BERT fine-tuned parameters have been saved to: {args.model_save_path}")

    elif args.task == 'test':
        # Load the model
        model_path = args.model_save_path  # Replace with the actual model save path
        model = torch.load(model_path)
        model.eval()
        test_path = args.root_path + 'test.json'
        with open(test_path, 'r', encoding='utf-8') as f:
            test_datas = json.load(f)
        print(f"Test Num:{len(test_datas)}")
        datas = evaluate(model, test_datas, slots_num, all_intents, tokenizer, device)
        save_json_file = './results/test_intent_slot_pred_bert.json'
        with open(save_json_file, 'w', encoding='utf-8') as file:
            json.dump(datas, file, ensure_ascii=False, indent=4)
        print(f'For each Query, the corresponding Intent and Slots (converted into keywords) predicted by BERT have been matched, and the file is saved at: {save_json_file }')


if __name__ == '__main__':
    main()





