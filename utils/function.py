import os, torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from itertools import combinations


def print_args(args):
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')
    print("=" * 50)
    

def plot_cluster_2D(args, data, labels, idx, block_idx=None):
    
    if block_idx is not None:
        save_path = os.path.join(args.save_tSNE_directory, f'{args.method}', 
                             f'{args.mode}', f'M{block_idx}_{idx}.pdf')
    else:
        save_path = os.path.join(args.save_tSNE_directory, f'{args.method}', 
                             f'{args.mode}', f'{idx}.pdf')
        
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
        
    cluster_num = args.label_num

    data = np.array(data)

    tsne_2D = TSNE(n_components=2, init='pca', random_state=42)
    data = tsne_2D.fit_transform(data)

    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig, ax = plt.subplots(figsize=(3, 3))
    cmap = plt.get_cmap('viridis')

    colors = [cmap(label / (cluster_num - 1)) for label in labels]
    plt.scatter(data[:, 0], data[:, 1], c=colors, cmap='viridis', s=2)

    plt.xticks([])
    plt.yticks([])

    ax.set_xlabel(args.method)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    
    
def save_model(args, model, block_idx=None):
    
    if block_idx is not None:
        save_path = os.path.join(args.save_model_directory, f'{args.method}', 
                             f'{args.mode}', f'M{block_idx}_best.pt')
    else:
        save_path = os.path.join(args.save_model_directory, f'{args.method}', 
                             f'{args.mode}', 'best.pt')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    torch.save(model, save_path)
    print('Saving model to {}'.format(save_path))
    

def load_model(args, block_idx=None):
    
    if block_idx is not None:
        load_path = os.path.join(args.save_model_directory, f'{args.method}', 
                             f'{args.mode}', f'M{block_idx}_best.pt')
    else:
        load_path = os.path.join(args.save_model_directory, f'{args.method}', 
                             f'{args.mode}', 'best.pt')
    
    assert os.path.exists(load_path)
    
    model = torch.load(load_path, map_location='cpu')
    model = model.cuda() if args.is_cuda else model

    print('\nLoading model from {}'.format(load_path))
    
    return model


def pairwise_sample(labels):
    
    assert labels is not None
    
    labels = labels.cpu().data.numpy()
    indices = np.arange(0, len(labels), 1)
    pairs = np.array(list(combinations(indices, 2)))
    
    pair_labels = (labels[pairs[:, 0]] == labels[pairs[:, 1]])

    pair_matrix = np.eye(len(labels))
    ind = np.where(pair_labels)
    
    pair_matrix[pairs[ind[0], 0], pairs[ind[0], 1]] = 1
    pair_matrix[pairs[ind[0], 1], pairs[ind[0], 0]] = 1

    return torch.LongTensor(pairs), torch.LongTensor(pair_labels.astype(int)),torch.LongTensor(pair_matrix)
