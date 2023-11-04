import sys, os
root_path = "/home/Xiaoyan/ReIMvent"
sys.path.append(root_path)

from utils.args_utils import get_args
from trainer import Trainer, IncreTrainer


if __name__ =='__main__':
    
    args = get_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu
    
    if args.mode == 'offline':
        trainer = Trainer(args)
        trainer.train_model()
        trainer.test_model()
        
    elif 'incremental' in args.mode:
        for block_idx in range (args.block_num):
            if block_idx == 0:
                trainer = Trainer(args, block_idx=block_idx)
                trainer.train_model(block_idx=block_idx)
                trainer.test_model(block_idx=block_idx)
            else:
                trainer = IncreTrainer(args, block_idx=block_idx)
                trainer.train_model(block_idx=block_idx)
                trainer.test_model(block_idx=block_idx)
        
    elif args.mode == 'from_scratch':
        for block_idx in range (args.block_num):
            trainer = Trainer(args, block_idx=block_idx)
            trainer.train_model(block_idx=block_idx)
            trainer.test_model(block_idx=block_idx)
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
