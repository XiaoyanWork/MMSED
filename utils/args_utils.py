import argparse

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', type=str, default='incremental_entropy', choices=['offline', 'from_scratch', 'incremental_random', 
                                                                        'incremental_entropy', 'incremental_wo_sample'])
    parser.add_argument('--method', type=str, default='ReIMvent', choices=['BERT', 'BERT_ResNet', 'CLIP', 'ViLT', 
                                                                           'DRMM', 'SSE_CBD', 'OWSEC', 'ReIMvent', 
                                                                           'ReIMvent_t', 'ReIMvent_k'])
    parser.add_argument('--ablation', type=str, default='idfd_cel', choices=['none', 'wo_rgs', 'wo_cf', 'wo_tri',
                                                                             'wo_idfd', 'cel', 'idfd_cel'])
    parser.add_argument('--tokenizer', type=str, default='CLIP', choices=['CLIP', 'BERT'])
    parser.add_argument('--block_num', type=int, default=9)
    
    parser.add_argument('--clip_directory', type=str, default='./pretrained_models/clip-transformer')
    parser.add_argument('--bert_directory', type=str, default='./pretrained_models/bert-base-uncased')
    parser.add_argument('--save_model_directory', type=str, default='./saved_model')
    parser.add_argument('--save_tSNE_directory', type=str, default='./tSNE')
    parser.add_argument('--data_root', type=str, default='./data')
    
    parser.add_argument('--max_img_num', type=int, default=4)
    parser.add_argument('--is_cuda', type=bool, default=True)
    parser.add_argument('--visible_gpu', type=str, default='2')
    parser.add_argument('--att_hidden_size', type=float, default=512)
    parser.add_argument('--att_heads_num', type=float, default=8)
    parser.add_argument('--label_num', type=int, default=-1)
    parser.add_argument('--epoch_num', type=int, default=1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)     
    parser.add_argument('--train_len', type=int, default=-1)
    parser.add_argument('--margin', type=float, default=0.6)
    parser.add_argument('--npc_in_dim', type=float, default=512)
    parser.add_argument('--npc_tau', type=float, default=1)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--npc_loss_tau', type=float, default=1)
    parser.add_argument('--norm_power', type=int, default=2)
    args = parser.parse_args()

    return args