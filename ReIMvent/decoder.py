import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        self.args = args
        self.linear_layer = nn.Linear(self.args.att_hidden_size, self.args.label_num)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, features):
        assert features is not None

        out = self.linear_layer(features)
        pred_out = self.softmax(out)

        return pred_out