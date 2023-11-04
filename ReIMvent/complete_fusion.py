import torch
import torch.nn as nn


class Com_Fusion(nn.Module):
    def __init__(self, args):
        super(Com_Fusion, self).__init__()
        self.args = args

        self.t2v_att_layer = nn.MultiheadAttention(embed_dim=self.args.att_hidden_size, num_heads=self.args.att_heads_num, batch_first=True)
        self.v2t_att_layer = nn.MultiheadAttention(embed_dim=self.args.att_hidden_size, num_heads=self.args.att_heads_num, batch_first=True)
    

    def forward(self, encoded_text_features, encoded_img_features, img_mask):
        assert encoded_text_features is not None
        assert encoded_img_features is not None
        assert img_mask is not None
        
        t2v_query = encoded_img_features
        t2v_key = encoded_text_features
        t2v_value = encoded_text_features
        t2v_attn_mask = 1 - img_mask

        t2v_output, _ = self.t2v_att_layer(query=t2v_query, 
                                           key=t2v_key, 
                                           value=t2v_value, 
                                           )
        
        t2v_output = t2v_output
        
        v2t_query = encoded_text_features
        v2t_key = t2v_output
        v2t_value = t2v_output
        v2t_key_mask = 1 - img_mask
        
        v2t_output, _ = self.v2t_att_layer(query=v2t_query, 
                                           key=v2t_key, 
                                           value=v2t_value, 
                                           key_padding_mask=v2t_key_mask.to(torch.float)
                                           )
        
        aggr_features = v2t_output

        return aggr_features