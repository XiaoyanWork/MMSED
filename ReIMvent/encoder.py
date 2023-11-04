import torch
import torch.nn as nn

from transformers import CLIPModel


class CLIP_Encoder(nn.Module):
    def __init__(self, args):
        super(CLIP_Encoder, self).__init__()
        
        self.args = args
        self.model = CLIPModel.from_pretrained(args.clip_directory)
            
        
    def forward(self, text_ids=None, img_datas=None, text_att_mask=None):
        
        assert text_ids is not None
        assert text_att_mask is not None

        text_outputs = self.model.text_model(input_ids=text_ids, attention_mask=text_att_mask)
        text_pooler_output = text_outputs[1]
        text_features = self.model.text_projection(text_pooler_output)
        
        encoded_text_features = text_features.unsqueeze(1)
        
        if self.args.method == 'ReIMvent_t':
            return encoded_text_features
        
        else:
            batch_size = img_datas.shape[0]
            batch_img_num = img_datas.shape[1]
            
            flatten_img_datas = torch.flatten(img_datas, start_dim=0, end_dim=1)
            vision_outputs = self.model.vision_model(pixel_values=flatten_img_datas)
            img_pooler_output = vision_outputs[1]  # pooler_output
            image_features = self.model.visual_projection(img_pooler_output)

            encoded_img_features = image_features.view(batch_size, batch_img_num, image_features.shape[-1])
        
            return encoded_text_features, encoded_img_features