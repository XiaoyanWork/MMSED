import torch
import torch.nn as nn

from ReIMvent.encoder import CLIP_Encoder
from ReIMvent.complete_fusion import Com_Fusion
from ReIMvent.rein_select import Agent
from ReIMvent.decoder import Decoder


class ReIMvent_Text_Model(nn.Module):
    def __init__(self, args):
        super(ReIMvent_Text_Model, self).__init__()
        self.args = args
        
        self.encoder = CLIP_Encoder(self.args)
        text_agent = Agent(self.args)
        self.aggregater = Com_Fusion(self.args)
            
        self.decoder = Decoder(self.args)

    def forward(self, 
                text_ids=None, 
                text_att_mask=None,
                img_datas=None, 
                img_mask=None
                ):
        
        encoded_text_features = self.encoder(text_ids, img_datas, text_att_mask)
        encoded_text_features = text_agent(encoded_text_features)
        pred_out = self.decoder(encoded_text_features)
        
        return aggr_features, pred_out