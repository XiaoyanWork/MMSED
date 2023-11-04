import torch, time, statistics
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.cluster import SpectralClustering, KMeans
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score

from data_utils.data import Data
from utils.get_model import get_model
from utils.incre_linear import Linear_for_Incre
from utils.function import print_args, plot_cluster_2D, save_model, load_model, pairwise_sample

from ReIMvent.idfd_loss import NonParametricClassifier, npc_Loss, Normalize


class Trainer(object):
    def __init__(self, args, block_idx=None):
        
        self.args = args
        
        self.data = Data(self.args)
        self.total_steps = len(self.get_train_loader(block_idx)) * self.args.epoch_num
        
        self.model = get_model(self.args)
        self.model = self.model.cuda() if self.args.is_cuda else self.model
        
        self.npc = NonParametricClassifier(self.args)
        self.npc = self.npc.cuda() if self.args.is_cuda else self.npc
        
        self.norm = Normalize(self.args)
        self.norm = self.norm.cuda() if self.args.is_cuda else self.norm
        
        self.cross_entropy = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=self.args.margin)
        self.cluster_friendly = npc_Loss(self.args)
        
        self.spec_clus = SpectralClustering(n_clusters=self.args.label_num, affinity='nearest_neighbors')
        self.kmeas = KMeans(n_clusters=self.args.label_num, n_init=10)
        
        self._initial_params()
        
    def _initial_params(self):
        self.epoch_time_consume = []
        self.global_steps = 0
        self.best_val_loss = 1e7
        self.patience = self.args.patience
        self.best_ep = 0
        
        assert self.model is not None
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                         num_warmup_steps=0.1*self.total_steps,
                                                         num_training_steps=self.total_steps)


    def get_train_loader(self, block_idx=None):     
        if self.data.dataset(flag='train') is None:
            self.data.load_dataset(flag='train', block_idx=block_idx)
        return self.data.get_dataloader(dataset=self.data.dataset(flag='train'))
    
    def get_valid_loader(self, block_idx=None):
        if self.data.dataset(flag='valid') is None:
            self.data.load_dataset(flag='valid', block_idx=block_idx)
        return self.data.get_dataloader(dataset=self.data.dataset(flag='valid'))
    
    def get_test_loader(self, block_idx=None):
        if self.data.dataset(flag='test') is None:
            self.data.load_dataset(flag='test', block_idx=block_idx)
        return self.data.get_dataloader(dataset=self.data.dataset(flag='test'))
    

    def train_model(self, block_idx=None):
        print_args(self.args)
        self.model.train()
        self.optimizer.zero_grad()

        for ep in range(self.args.epoch_num):
            
            epoch_begin_time = time.time()
            
            train_acc_instance, train_loss = self.train_step(block_idx)
            valid_acc_instance, valid_loss = self.valid_model(block_idx)
            
            epoch_consume_time = time.time() - epoch_begin_time
            
            print('=' * 50)
            if block_idx is not None:
                print('Block {:d}   '.format(block_idx))
            print('Epoch {:d}   '
                  'Epoch Time {:5.2f}s | '
                  'Train Loss {:5.2f} | '
                  'Train ACC {:5.2f}% | '
                  'Valid Loss {:5.2f} | '
                  'Valid ACC {:5.2f}%   '.format(ep, epoch_consume_time, train_loss, train_acc_instance, valid_loss, valid_acc_instance))
            print("=" * 50)

            if valid_loss < self.best_val_loss:
                self.best_val_loss = valid_loss
                self.best_ep = ep
                self.patience = self.args.patience
                save_model(self.args, self.model, block_idx=block_idx)
            else:
                self.patience -= 1

            if self.patience <= 0:
                print("Run out of patience. BREAK!")
                break
    
        print('*' * 50)
        print('Time consumed first epoch: {:5.2f}'.format(self.epoch_time_consume[0]))
        print('Time consumed per subsequently epoch: {:5.2f}'.format(statistics.mean(self.epoch_time_consume[1:])))
        print('Time consumed per epoch: {:5.2f}'.format(statistics.mean(self.epoch_time_consume)))
        print('*' * 50)


    def train_step(self, block_idx=None):
        train_loader = self.get_train_loader(block_idx)
        train_acc_instance, train_loss = 0, 0
        train_total_instance = len(train_loader.dataset.datas)

        for step, batch in enumerate(train_loader):
            
            self.global_steps += 1

            batch_text_ids = batch['text_ids'].cuda() if self.args.is_cuda else batch['title_ids']
            batch_text_att_mask = batch['text_att_mask'].cuda() if self.args.is_cuda else batch['text_att_mask']
            batch_img_datas = batch['img_datas'].cuda() if self.args.is_cuda else batch['img_datas']
            batch_img_mask = batch['img_mask'].cuda() if self.args.is_cuda else batch['img_mask']
            batch_label_ids = batch['label_ids'].cuda() if self.args.is_cuda else batch['label_ids']

            batch_feature_out, batch_pred_out = self.model(batch_text_ids, 
                                                batch_text_att_mask, 
                                                batch_img_datas, 
                                                batch_img_mask)

            batch_pred_out = batch_pred_out.squeeze(axis=1)
            batch_feature_out = batch_feature_out.squeeze(axis=1)
            _, batch_pred = torch.max(batch_pred_out, dim=-1)
            
            if 'ReIMvent' in self.args.method:
                triplet_loss = self.triplet_loss_compute(batch_label_ids, batch_feature_out)
                idfd_loss = self.clu_friendly_compute(batch_feature_out)
                if self.args.ablation in ['none', 'wo_rgs', 'wo_cf']:
                    batch_loss = triplet_loss + idfd_loss
                elif self.args.ablation in ['wo_tri', 'idfd_cel']:
                    batch_loss = idfd_loss
                elif self.args.ablation == 'wo_idfd':
                    batch_loss = triplet_loss
                    
            if 'ReIMvent' not in self.args.method or 'cel' in self.args.ablation:
                batch_loss = self.cross_entropy(batch_pred_out, batch_label_ids)
                if self.args.ablation == 'idfd_cel':
                    batch_loss += idfd_loss
                
            train_acc_instance += torch.sum(batch_pred == batch_label_ids).int().item()
            train_acc = (train_acc_instance / train_total_instance) * 100
            
            batch_loss = batch_loss / self.args.gradient_accumulation_steps
            train_loss += batch_loss.item()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_grad_norm)
            batch_loss.backward()

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        if (step + 1) % self.args.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        train_loss = (train_loss * self.args.gradient_accumulation_steps) / (step + 1)
        
        return train_acc, train_loss


    def valid_model(self, block_idx=None):
        self.model.eval()
        
        valid_loader = self.get_valid_loader(block_idx)
        valid_acc_instance, valid_loss = 0, 0
        valid_total_instance = len(valid_loader.dataset.datas)
        
        with torch.no_grad():
            for step, batch in enumerate(valid_loader):
                
                batch_text_ids = batch['text_ids'].cuda() if self.args.is_cuda else batch['title_ids']
                batch_text_att_mask = batch['text_att_mask'].cuda() if self.args.is_cuda else batch['text_att_mask']
                batch_img_datas = batch['img_datas'].cuda() if self.args.is_cuda else batch['img_datas']
                batch_img_mask = batch['img_mask'].cuda() if self.args.is_cuda else batch['img_mask']
                batch_label_ids = batch['label_ids'].cuda() if self.args.is_cuda else batch['label_ids']
                
                batch_feature_out, batch_pred_out = self.model(batch_text_ids, 
                                                    batch_text_att_mask, 
                                                    batch_img_datas, 
                                                    batch_img_mask)

                batch_pred_out = batch_pred_out.squeeze(axis=1)
                batch_feature_out = batch_feature_out.squeeze(axis=1)
                _, pred = torch.max(batch_pred_out, dim=-1)
                
                if 'ReIMvent' in self.args.method:
                    triplet_loss = self.triplet_loss_compute(batch_label_ids, batch_feature_out)
                    idfd_loss = self.clu_friendly_compute(batch_feature_out)
                    if self.args.ablation in ['none', 'wo_rgs', 'wo_cf']:
                        batch_loss = triplet_loss + idfd_loss
                    elif self.args.ablation in ['wo_tri', 'idfd_cel']:
                        batch_loss = idfd_loss
                    elif self.args.ablation == 'wo_idfd':
                        batch_loss = triplet_loss
                        
                if 'ReIMvent' not in self.args.method or 'cel' in self.args.ablation:
                    batch_loss = self.cross_entropy(batch_pred_out, batch_label_ids)
                    if self.args.ablation == 'idfd_cel':
                        batch_loss += idfd_loss

                valid_acc_instance += torch.sum(pred == batch_label_ids).int().item()
                valid_acc = (valid_acc_instance / valid_total_instance) * 100
                
                valid_loss += batch_loss.item()
            
        valid_loss /= (step + 1)
        
        self.model.train()
        return valid_acc, valid_loss


    def test_model(self, block_idx=None):
        
        test_NMI, test_AMI, test_ARI = [], [], []
        
        for i in range(5):
            test_NMI_value, test_AMI_value, test_ARI_value = self.test_step(idx=i, block_idx=block_idx)
            test_NMI.append(test_NMI_value)
            test_AMI.append(test_AMI_value)
            test_ARI.append(test_ARI_value)
        
        test_NMI_mean = statistics.mean(test_NMI)
        test_NMI_stdev = statistics.stdev(test_NMI)
        test_AMI_mean = statistics.mean(test_AMI)
        test_AMI_stdev = statistics.stdev(test_AMI)                
        test_ARI_mean = statistics.mean(test_ARI)
        test_ARI_stdev = statistics.stdev(test_ARI)
        
        print('\nFinal testing results:')
        print('-' * 50)
        print('NMI mean  {:5.2f} | '
              'NMI stdev {:5.2f} | '
              'AMI mean  {:5.2f} | '
              'AMI stdev {:5.2f} | '
              'ARI mean  {:5.2f} | '
              'ARI stdev {:5.2f}   '.format(test_NMI_mean, test_NMI_stdev, test_AMI_mean, test_AMI_stdev, test_ARI_mean, test_ARI_stdev))
        print('-' * 50)    


    def test_step(self, idx, block_idx=None):
        self.model = load_model(self.args, block_idx=block_idx)
        self.model.eval()
                
        test_loader = self.get_test_loader(block_idx)
        test_acc_instance, test_loss = 0, 0
        test_total_instance = len(test_loader.dataset.datas)
        all_data, all_label = [], []

        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                
                batch_text_ids = batch['text_ids'].cuda() if self.args.is_cuda else batch['title_ids']
                batch_text_att_mask = batch['text_att_mask'].cuda() if self.args.is_cuda else batch['text_att_mask']
                batch_img_datas = batch['img_datas'].cuda() if self.args.is_cuda else batch['img_datas']
                batch_img_mask = batch['img_mask'].cuda() if self.args.is_cuda else batch['img_mask']
                batch_label_ids = batch['label_ids'].cuda() if self.args.is_cuda else batch['label_ids']

                batch_feature_out, batch_pred_out = self.model(batch_text_ids, 
                                                    batch_text_att_mask, 
                                                    batch_img_datas, 
                                                    batch_img_mask)

                batch_pred_out = batch_pred_out.squeeze(axis=1)
                batch_feature_out = batch_feature_out.squeeze(axis=1)
                _, pred = torch.max(batch_pred_out, dim=-1)
                
                if 'ReIMvent' in self.args.method:
                    triplet_loss = self.triplet_loss_compute(batch_label_ids, batch_feature_out)
                    idfd_loss = self.clu_friendly_compute(batch_feature_out)
                    if self.args.ablation in ['none', 'wo_rgs', 'wo_cf']:
                        batch_loss = triplet_loss + idfd_loss
                    elif self.args.ablation in ['wo_tri', 'idfd_cel']:
                        batch_loss = idfd_loss
                    elif self.args.ablation == 'wo_idfd':
                        batch_loss = triplet_loss
                        
                if 'ReIMvent' not in self.args.method or 'cel' in self.args.ablation:
                    batch_loss = self.cross_entropy(batch_pred_out, batch_label_ids)
                    if self.args.ablation == 'idfd_cel':
                        batch_loss += idfd_loss

                test_acc_instance += torch.sum(pred == batch_label_ids).int().item()
                test_acc = (test_acc_instance / test_total_instance) * 100
                
                test_loss += batch_loss.item()
                
                all_data.extend(batch_feature_out.tolist())
                all_label.extend(batch_label_ids.tolist())
        
        if args.clustering_method == 'kmeans':
            cluster_pred = self.kmeas.fit_predict(all_data)
        elif args.clustering_method == 'spectral':
            cluster_pred = self.spec_clus.fit_predict(all_data)
        else:
            raise ValueError(f"Invalid clustering method: {args.clustering_method}")
            
        NMI_value = normalized_mutual_info_score(all_label, cluster_pred)
        AMI_value = adjusted_mutual_info_score(all_label, cluster_pred)
        ARI_value = adjusted_rand_score(all_label, cluster_pred)
        
        plot_cluster_2D(self.args, all_data, cluster_pred, idx, block_idx=block_idx)
            
        test_loss /= (step + 1)
        
        return NMI_value, AMI_value, ARI_value

    
    def triplet_loss_compute(self, batch_label_ids, batch_feature_out):
        triplets_loss, triplets = 0, 0
        pairs, pair_labels, pair_matrix = pairwise_sample(batch_label_ids)
        
        for i, anchor in enumerate(batch_feature_out):
            pos_indices = np.where((pairs[:, 0] == i) & pair_labels)[0]
            neg_indices = np.where((pairs[:, 0] == i) & ~pair_labels)[0]
            
            if len(pos_indices) > 0 and len(neg_indices) > 0:
                pos_idx = np.random.choice(pos_indices)
                neg_idx = np.random.choice(neg_indices)
                
                positive = batch_feature_out[pairs[pos_idx, 1]]
                negative = batch_feature_out[pairs[neg_idx, 1]]
                
                triplet_loss = self.triplet_loss(anchor, positive, negative)
                triplets_loss += triplet_loss
                triplets += 1
        
        if triplets == 0:
            triplets_loss = torch.tensor(0.0, requires_grad=True, device=batch_feature_out.device)
        else:
            triplets_loss /= triplets
            
        return triplets_loss
    
    
    def clu_friendly_compute(self, batch_feature_out):
        batch_feature_out = self.norm(batch_feature_out)
        
        indexes = torch.arange(batch_feature_out.shape[0])
        indexes = indexes.cuda() if self.args.is_cuda else indexes
        npc_out = self.npc(batch_feature_out, indexes)
        
        loss_id, loss_fd = self.cluster_friendly(npc_out, batch_feature_out, indexes)   
        npc_loss = loss_id + loss_fd
        
        return npc_loss

class IncreTrainer(Trainer):
    def __init__(self, args, block_idx):
        super(IncreTrainer, self).__init__(args, block_idx)
        
        self.args = args
        
        self.model = load_model(args, block_idx=block_idx - 1)
        self.model = Linear_for_Incre(self.args, self.model)
        self.model = self.model.cuda() if self.args.is_cuda else self.model