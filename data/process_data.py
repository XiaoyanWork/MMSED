import os, csv, json
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.48, 0.498, 0.531),
                         (0.214, 0.207, 0.207))]
)

data_path = ''
text_path = os.path.join(data_path, 'annotations')


def merge_key_data(data):    
    result = {}
    
    for item in data:
        main_key = item.pop('tweet_id')
        
        if main_key not in result:
            result[main_key] = {k: [v] if k not in ['tweet_text', 'label', 'date'] else v for k, v in item.items()}
    
        else:
            for key, value in item.items():
                if key in ['tweet_text', 'label', 'date'] and value != result[main_key][key]:
                    raise ValueError(f"Key {key} values are not the same")
                elif key not in ['tweet_text', 'label', 'date']:
                    result[main_key][key].append(value)
    
    final_result = [{'tweet_id': [k], **v} for k, v in result.items()]
    
    return final_result


def get_tsv_datas(data_path, label):
    with open(data_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        data = [{**row, 'label': label} for row in reader]
    
    for dat in data:
        dat['date'] = dat['image_path'].split('/')[2]
        
    data = merge_key_data(data)
    
    return data


def load_img(img_path):
    image = Image.open(img_path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = transform(image)
    img_dat = np.array(image)

    return img_dat


def get_img_data(datas, root_path):
    
    all_img_data = {}
    
    for data in tqdm(datas):
        for i in data['image_path']:
            img_path = os.path.join(root_path, i)
            img_data = load_img(img_path)
            all_img_data[i] = img_data

    return all_img_data


def save_json(data_list):
    with open('./data/all_datas.json', 'w') as fw:
        json.dump(data_list, fw)
        

def save_npy(img_dict):
    np.save('./data/all_img_datas.npy', img_dict)


if __name__ == '__main__':
    all_data = []
    
    tsv_list = os.listdir(text_path)
    
    for k, path in enumerate(tsv_list):
        label = path[:-15]
        if 'DS_Store' in path:
            continue
        data = get_tsv_datas(os.path.join(text_path, path), label)
        for item in data:
            all_data.append(item)
    
    img_dict = get_img_data(all_data, data_path)
    
    save_npy(img_dict)
    save_json(all_data)
