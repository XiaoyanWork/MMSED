import json, random, copy, os, math
import numpy as np

from datetime import datetime
from collections import Counter

root_path = './data'
text_path = os.path.join(root_path, 'all_datas.json')
image_path = os.path.join(root_path, 'all_img_datas.npy')
    

def get_json(path):
    with open(path, 'r') as f:
        data_list = json.load(f)
    return data_list


def get_npy(path):
    img_datas = np.load(path, allow_pickle=True)
    img_datas = img_datas.tolist()
    return img_datas


def custom_sort(date_str):
    return datetime.strptime(date_str, '%d_%m_%Y')


def split_blocks(data_list):
    all_date = set([i['date'] for i in data_list])
    sorted_date = sorted(all_date, key=custom_sort)
    
    blocks = []
    start_index = 0
    
    for blk_idx, end_index in enumerate(range(41, 98, 7)):
        blk_name = 'M{}'.format(blk_idx)
        block = [data for data in data_list if data['date'] in sorted_date[start_index:end_index]]
        from_scratch_block = [data for data in data_list if data['date'] in sorted_date[:end_index]]
        start_index = end_index
        
        block_result = {'block_name': blk_name,
                        'all_current_data': from_scratch_block,
                        'block_data': block}
        
        blocks.append(block_result)
    
    return blocks


def incre_random_split(blocks):
        
    for blk_idx, blk in enumerate(blocks):
        from_scratch_train = []
        all_labels = set([i['label'] for i in blk['block_data']])
        
        train_data, valid_data, test_data = [], [], []

        for label in all_labels:
            label_data = [d for d in blk['block_data'] if d['label'] == label]
            random.shuffle(label_data)
            
            idx1 = int(len(label_data) * 0.8)
            idx2 = int(len(label_data) * 0.9)

            train_data += label_data[:idx1]
            valid_data += label_data[idx1:idx2]
            test_data += label_data[idx2:]
        
        if blk_idx != 0:
            potential_train_data = []
            valid_data += prev_valid_data
            test_data += prev_test_data
            
            for data in blk['all_current_data']:
                if data not in valid_data+test_data+train_data:
                    potential_train_data.append(data)
            
            random.shuffle(potential_train_data)
            
            train_data += potential_train_data[:len(train_data)]
        
        prev_valid_data = copy.deepcopy(valid_data)
        prev_test_data = copy.deepcopy(test_data)
        
        for data in blk['all_current_data']:
            if data not in valid_data+test_data:
                from_scratch_train.append(data)
        
        block_result = {'train_random': train_data,
                        'train_from_scratch': from_scratch_train,
                        'valid': valid_data,
                        'test': test_data}
        blocks[blk_idx].update(block_result)
        
    return blocks


def compute_text_list_entropy(text_datas):
        
    word_counts = Counter()
    for sentence in text_datas:
        words = sentence.split()
        word_counts.update(words)

    total_words = sum(word_counts.values())
    word_probabilities = {word: count / total_words for word, count in word_counts.items()}

    entropies = []
    for sentence in text_datas:
        words = sentence.split()
        sentence_entropy = 0
        for word in words:
            word_probability = word_probabilities[word]
            sentence_entropy -= word_probability * math.log2(word_probability)
        entropies.append(sentence_entropy)

    return entropies


def incre_entropy_split(blocks):
        
    for blk_idx, blk in enumerate(blocks):
        potential_train, train_data = [], []
        
        for data in blk['block_data']:
            if data not in blk['valid']+blk['test']:
                train_data.append(data)
                
        for data in blk['all_current_data']:
            if data not in blk['valid']+blk['test']+train_data:
                potential_train.append(data)
        
        potential_train_text = [data['tweet_text'] for data in potential_train]
        
        potential_train_entropy = compute_text_list_entropy(potential_train_text)
        sorted_potential_train = sorted(potential_train, key=lambda x: potential_train_entropy[potential_train_text.index(x['tweet_text'])], reverse=True)
        
        train_data += sorted_potential_train[:len(train_data)]
        
        blocks[blk_idx]['train_entropy'] = train_data
        
    return blocks


def incre_wo_sample(data_blocks):
    
    for blk_idx, blk in enumerate(blocks):
        train_wo_sample_data = []
        for data in blk['block_data']:
            if data not in blk['valid']+blk['test']:
                train_wo_sample_data.append(data)
        
        blocks[blk_idx]['train_wo_sample'] = train_wo_sample_data
    
    return blocks


def save_json(data_list, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as fw:
        json.dump(data_list, fw)


def save_block_jsons(data_blocks):
    random_path = os.path.join(root_path, 'incremental/random')
    for blk_idx, blk in enumerate(data_blocks):
        block_name = blk['block_name']
        for data_type, data in zip(['train', 'valid', 'test'],
                                   [blk['train_random'], blk['valid'], blk['test']]):
            file_name = f'{block_name}_{data_type}.json'
            save_path = os.path.join(random_path, file_name)
            save_json(data, save_path)
    
    entropy_path = os.path.join(root_path, 'incremental/entropy')
    for blk_idx, blk in enumerate(data_blocks):
        block_name = blk['block_name']
        for data_type, data in zip(['train', 'valid', 'test'],
                                   [blk['train_entropy'], blk['valid'], blk['test']]):
            file_name = f'{block_name}_{data_type}.json'
            save_path = os.path.join(entropy_path, file_name)
            save_json(data, save_path)
    
    from_scratch_path = os.path.join(root_path, 'from_scratch')
    for blk_idx, blk in enumerate(data_blocks):
        block_name = blk['block_name']
        for data_type, data in zip(['train', 'valid', 'test'],
                                   [blk['train_from_scratch'], blk['valid'], blk['test']]):
            file_name = f'{block_name}_{data_type}.json'
            save_path = os.path.join(from_scratch_path, file_name)
            save_json(data, save_path)
    
    wo_sample_path = os.path.join(root_path, 'incremental/wo_sample')
    for blk_idx, blk in enumerate(data_blocks):
        block_name = blk['block_name']
        for data_type, data in zip(['train', 'valid', 'test'],
                                   [blk['train_wo_sample'], blk['valid'], blk['test']]):
            file_name = f'{block_name}_{data_type}.json'
            save_path = os.path.join(wo_sample_path, file_name)
            save_json(data, save_path)
    
    offline_path = os.path.join(root_path, 'offline')
    for data_type, data in zip(['train', 'valid', 'test'],
                                [blocks[8]['train_from_scratch'], 
                                blocks[8]['valid'], 
                                blocks[8]['test']]):
        file_name = f'{data_type}.json'
        save_path = os.path.join(offline_path, file_name)
        save_json(data, save_path)


if __name__ == '__main__':
    data_list = get_json(text_path)
    all_img = get_npy(image_path)
    
    blocks = split_blocks(data_list)
    
    blocks = incre_random_split(blocks)
    
    blocks = incre_entropy_split(blocks)
    
    blocks = incre_wo_sample(blocks)
    
    blocks
    
    save_block_jsons(blocks)    
    
    '''
        Here's an example of getting image data:
            img_data = {}
            img_data.update({k: all_img[k] for i in range(len(data_list)) for k in data_list[i]['image_path']})
    '''