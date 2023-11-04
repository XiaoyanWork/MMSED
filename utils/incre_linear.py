import torch.nn as nn


class Linear_for_Incre(nn.Module):
    def __init__(self, args, model):
        super(Linear_for_Incre, self).__init__()
        
        self.args = args
        self.model = model
        self.linear_for_incre = nn.Linear(self.args.att_hidden_size, args.label_num)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, 
                text_ids=None, 
                text_att_mask=None,
                img_datas=None, 
                img_mask=None):
        
        out_features, pred = self.model(text_ids, text_att_mask, img_datas, img_mask)
        
        out = self.linear_for_incre(out_features)
        pred_out = self.softmax(out)
        
        return out_features, pred_out