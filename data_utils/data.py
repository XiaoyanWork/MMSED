import torch
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, CLIPTokenizer

from data_utils.dataset import Datasets
from data_utils.data_funtion import padding_img_datas


class Data(object):
    def __init__(self, args):
        
        self.args = args
        
        if self.args.tokenizer == 'BERT':
             self.tokenizer = BertTokenizer.from_pretrained(args.bert_directory)
        elif self.args.tokenizer == 'CLIP':
            self.tokenizer = CLIPTokenizer.from_pretrained(args.clip_directory)
        else:
            raise ValueError(f"Invalid tokenizer: {self.args.tokenizer}")
        
        self.datasets = {}
        self.dataset_to_iter = {}


    def load_dataset(self, flag, block_idx=None):
        assert flag in ['train', 'valid', 'test']
        self.datasets[flag] = Datasets(self.args, self.tokenizer, flag, block_idx)


    def dataset(self, flag):
        assert flag in ['train', 'valid', 'test']
        return self.datasets.get(flag, None)
    

    def get_dataloader(self, dataset):
        if dataset in self.dataset_to_iter:
            return self.dataset_to_iter[dataset]
        else:
            assert isinstance(dataset, Dataset)
            self.dataset_to_iter[dataset] = DataLoader(dataset=dataset,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       collate_fn=self.collate_fn)
            return self.dataset_to_iter[dataset]
    
            
    def collate_fn(self, batch_datas):
        
        batch_text_ids = [b['text_ids'] for b in batch_datas]
        batch_text_att_mask = [b['text_att_mask'] for b in batch_datas]
        batch_img_datas = [b['img_datas'] for b in batch_datas]
        
        batch_labels = [b['label'] for b in batch_datas]
        batch_label_ids = [self.datasets['train'].label_dict.get(l) for l in batch_labels]
        
        batch_max_text_len = max([len(i) for i in batch_text_ids])

        padded_batch_text_ids = [b + [0 for _ in range(batch_max_text_len - len(b))] for b in batch_text_ids]
        padded_batch_text_att_mask = [b + [0 for _ in range(batch_max_text_len - len(b))] for b in batch_text_att_mask]
        
        padded_img_datas, img_mask = padding_img_datas(self.args, batch_img_datas)
        
        padded_batch_text_ids = torch.tensor(padded_batch_text_ids, dtype=torch.int64)
        padded_batch_text_att_mask = torch.tensor(padded_batch_text_att_mask, dtype=torch.int64)
        padded_img_datas = torch.tensor(np.array(padded_img_datas))
        img_mask = torch.tensor(img_mask, dtype=torch.int64)
        batch_label_ids = torch.tensor(batch_label_ids, dtype=torch.int64)

        padded_batch_datas = {
            'text_ids': padded_batch_text_ids,
            'text_att_mask': padded_batch_text_att_mask,
            'img_datas': padded_img_datas,
            'img_mask': img_mask,
            'label_ids': batch_label_ids
        }
        
        return padded_batch_datas