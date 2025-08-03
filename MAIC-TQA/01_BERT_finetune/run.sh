#!/bin/bash

# Execute training task
#python main.py --task='train' --bert_path='your_path' --epochs=5 --lr=1e-6 --batch_size=32
python main.py --task='train' --epochs=5 --lr=1e-5 --batch_size=16
# Execute testing task
#python main.py --task='test' --bert_path='your_path' --epochs=5 --lr=1e-6 --batch_size=32
python main.py --task='test'