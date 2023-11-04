import torch
import torch.nn as nn

from ReIMvent.encoder import CLIP_Encoder
from ReIMvent.complete_fusion import Com_Fusion
from ReIMvent.cross_att import Attention
from ReIMvent.decoder import Decoder
from ReIMvent.rein_select import Agent


class ReIMvent_Model(nn.Module):
    def __init__(self, args):
        super(ReIMvent_Model, self).__init__()
        self.args = args
        
        self.encoder = CLIP_Encoder(self.args)
        if self.args.ablation != 'wo_rgs':
            text_agent = Agent(self.args)
            image_agent = Agent(self.args)
        if self.args.ablation == 'wo_cf':
            self.aggregater = Attention(self.args)
        else:
            self.aggregater = Com_Fusion(self.args)
            
        self.decoder = Decoder(self.args)

    def forward(self, 
                text_ids=None, 
                text_att_mask=None,
                img_datas=None, 
                img_mask=None
                ):
        
        encoded_text_features, encoded_img_features = self.encoder(text_ids, img_datas, text_att_mask)
        if self.args.ablation != 'wo_rgs':
            encoded_text_features = text_agent(encoded_text_features)
            encoded_img_features = image_agent(encoded_img_features)
        aggr_features = self.aggregater(encoded_text_features, encoded_img_features, img_mask)
        
        pred_out = self.decoder(aggr_features)
        
        return aggr_features, pred_out