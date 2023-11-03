import os

from torch.utils.data.dataset import Dataset

from data_utils.data_funtion import get_npy, get_data, tokenize_all_datas, get_label_dict


class Datasets(Dataset):
    def __init__(self, args, tokenizer, flag, block_idx=None):

        self.args = args
        self.tokenizer = tokenizer
        
        all_img = get_npy(os.path.join(self.args.data_root, 'all_img_datas.npy'))
        
        if self.args.mode == 'offline':
            file_path = os.path.join(self.args.data_root, self.args.mode)
            file_path = os.path.join(file_path, f'{flag}.json')
        
        elif 'incremental' in self.args.mode:
            file_path = os.path.join(self.args.data_root, self.args.mode.replace("_", "/", 1))
            file_path = os.path.join(file_path, f'M{block_idx}_{flag}.json')
        
        elif self.args.mode == 'from_scratch':
            file_path = os.path.join(self.args.data_root, self.args.mode) 
            file_path = os.path.join(file_path, f'M{block_idx}_{flag}.json') 
        
        else:
            raise ValueError(f"Invalid mode: {self.args.mode}")

        self.datas, self.img_datas = get_data(file_path, all_img)

        self.tokenzied_all_datas = tokenize_all_datas(self.datas, self.tokenizer)

        if flag == 'train':
            self.label_dict, self.label_num, train_len = get_label_dict(self.datas)
            self.args.label_num = self.label_num
            self.args.train_len = train_len
            
        else:
            self.label_dict, self.label_num = None, None
        

    def __len__(self):
        return len(self.datas)


    def __getitem__(self, item):
        
        tmp_item = self.datas[item]
        
        tmp_text_ids = tmp_item['input_ids']
        tmp_text_att_mask = tmp_item['text_att_mask']
        tmp_label = tmp_item['label']
        tmp_img_datas = [self.img_datas.get(path) for path in tmp_item['image_path']]
        
        tmp = {
            'text_ids': tmp_text_ids, 
            'text_att_mask': tmp_text_att_mask, 
            'img_datas': tmp_img_datas, 
            'label': tmp_label
        }
        return tmp