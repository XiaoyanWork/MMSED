from baselines.BERT import BERT_Model
from baselines.BERT_ResNet import BERT_ResNet_Model
from baselines.CLIP import CLIP_Model
from baselines.DRMM import DRMM_Model
from baselines.OWSEC import OWSEC_Model
from baselines.SSECBD import SSE_CBD_Model
from baselines.ViLT import ViLT_Model

from ReIMvent.reimvent_model import ReIMvent_Model


def get_model(args):
    if args.method == 'BERT':
        model = BERT_Model(args)
    elif args.method == 'BERT_ResNet':
        model = BERT_ResNet_Model(args)
    elif args.method == 'CLIP':
        model = CLIP_Model(args)
    elif args.method == 'DRMM':
        model = DRMM_Model(args)
    elif args.method == 'SSE_CBD':
        model = SSE_CBD_Model(args)
    elif args.method == 'OWSEC':
        model = OWSEC_Model(args)
    elif args.method == 'ViLT':
        model = ViLT_Model(args)
    elif args.method in ['ReIMvent', 'ReIMvent_k']:
        model = ReIMvent_Model(args)
    elif args.method == 'ReIMvent_t':
        model = ReIMvent_Text_Model(args)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    return model
