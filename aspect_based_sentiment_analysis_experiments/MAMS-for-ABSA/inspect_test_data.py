import os
from torch.utils.data import DataLoader
from data_process.dataset import ABSADataset
import yaml
import os
from train.test import test

config = yaml.safe_load(open('config.yml'))

input_list = {
    'recurrent_capsnet': ['context', 'aspect'],
    'bert_capsnet': ['bert_token', 'bert_segment']
}

mode = config['mode']

base_path = config['base_path']
test_path = os.path.join(base_path, 'processed/test.npz')
test_data = ABSADataset(test_path, input_list[config['aspect_term_model']['type']])
config = config['aspect_term_model'][config['aspect_term_model']['type']]

# print(test_data.data.keys())
# print(test_data.len)
# print(test_data.input_list)
# dict_keys(['sentence', 'aspect', 'label', 'context', 'bert_token', 'bert_segment', 'td_left', 'td_right'])
# 1336
# ['bert_token', 'bert_segment']



test_loader = DataLoader(
    dataset=test_data,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)


for data in test_loader:
    input0, input1, label = data
    print('input0:', input0)
    print('input1:', input1)
    print('label', label)