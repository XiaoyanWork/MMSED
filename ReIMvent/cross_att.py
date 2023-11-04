import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args

        self.t2v_att_layer = nn.MultiheadAttention(embed_dim=self.args.att_hidden_size, num_heads=self.args.att_heads_num, batch_first=True)
    
    
    def forward(self, encoded_text_features, encoded_img_features, img_mask):
        assert encoded_text_features is not None
        assert encoded_img_features is not None
        assert img_mask is not None
        
        query = encoded_text_features           
        key = encoded_img_features
        value = encoded_img_features
        key_mask = 1 - img_mask

        v2t_output, _ = self.t2v_att_layer(query=query, 
                                           key=key, 
                                           value=value, 
                                           key_padding_mask=key_mask.to(torch.float)
                                           )
        
        aggr_features = v2t_output

        return aggr_features
