import os
import pickle

import torch

from train import make_aspect_term_model, make_aspect_category_model
from train.eval_ty import eval
from torch.utils.data import DataLoader
from data_process.dataset import ABSADatasetTY

filename = '0a5c0a4c-36f7-46c4-9f13-91f52ba45ea5'
input_test_dir = '/home/yiyi/Documents/masterthesis/CPD/data/ABSA/processed/'
output_test_dir = '/home/yiyi/Documents/masterthesis/CPD/data/ABSA/eval/'
output_test_path = os.path.join(output_test_dir, filename)


def test(config):
    mode = config['mode']
    if mode == 'term':
        model = make_aspect_term_model.make_model(config)
    else:
        model = make_aspect_category_model.make_model(config)
    model = model.cuda()
    model_path = os.path.join(config['base_path'], 'checkpoints/%s.pth' % config['aspect_' + mode + '_model']['type'])
    model.load_state_dict(torch.load(model_path))
    if mode == 'term':
        # test_loader = make_term_test_data(config)
        model_type = config['aspect_term_model']['type']
        if 'bert' in model_type:
            i_list = ['bert_token', 'bert_segment']
        else:
            i_list = ['sentence', 'aspect']

        base_path = config['base_path']
        ### TODO
        test_path = os.path.join(input_test_dir, filename + '.npz')
        test_data = ABSADatasetTY(test_path, i_list)
        config = config['aspect_term_model'][config['aspect_term_model']['type']]
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True
        )

    else:
        model_type = config['aspect_category_model']['type']
        if 'bert' in model_type:
            i_list = ['bert_token', 'bert_segment']
        else:
            i_list = ['sentence', 'aspect']
        base_path = config['base_path']
        test_path = os.path.join(base_path, 'processed/test_ty.npz')
        test_data = ABSADatasetTY(test_path, i_list)
        config = config['aspect_category_model'][config['aspect_category_model']['type']]
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True
        )
    predictions = eval(model, test_loader)
    print(len(predictions))

    with open(output_test_path, 'wb')as file:
        pickle.dump(predictions, file)
