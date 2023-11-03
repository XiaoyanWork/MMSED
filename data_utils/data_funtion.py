import os, json
import numpy as np

def get_npy(path):
    img_datas = np.load(path, allow_pickle=True)
    img_datas = img_datas.tolist()
    return img_datas


def get_data(file_path, all_img):
    assert os.path.exists(file_path), '[ASSERT]: Json data does not exist.'
    
    with open(file_path, 'r') as f:
        datas = json.load(f)
    
    img_datas = {}
    img_datas.update({k: all_img[k] for i in range(len(datas)) for k in datas[i]['image_path']})

    return datas, img_datas


def tokenize_all_datas(datas, tokenizer):
    tokenized_all_datas = []

    for data in datas:
        raw_text = data['tweet_text']
        text_token = tokenizer(raw_text, max_length=77, truncation=True)

        data['input_ids'] = text_token.data['input_ids']
        data['text_att_mask'] = text_token.data['attention_mask']

        tokenized_all_datas.append(data)

    return tokenized_all_datas


def get_label_dict(datas):
    set_labels = set([i['label'] for i in datas])
    label_num = len(set_labels)
    label_dict = {}
    
    for i, label in enumerate(sorted(set_labels)):
        label_dict[label] = i
    
    return label_dict, label_num, len(datas)


def padding_img_datas(args, batch_img_datas):

    assert max([len(b) for b in batch_img_datas]) <= args.max_img_num, 'Number of images in batch exceeds max image number.'

    padding_matrix = np.zeros_like(batch_img_datas[0][0])

    padded_img_datas, img_mask = [], []

    for img_data in batch_img_datas:
        tmp_data = img_data + [padding_matrix for _ in range(args.max_img_num - len(img_data))]
        tmp_mask = len(img_data) * [1] + [0 for _ in range(args.max_img_num - len(img_data))]

        padded_img_datas.append(tmp_data)
        img_mask.append(tmp_mask)

    return padded_img_datas, img_mask