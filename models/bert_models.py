from torch import nn
from transformers import BertModel
import os
from torchcrf import CRF
import torch

class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)

        self.bert_config = self.bert_module.config




    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)


class BertCRF(BaseModel):

    def __init__(self,
                 bert_dir,
                 num_tags,
                 dropout_prob=0.1,
                 **kwargs):
        super(CRFModel, self).__init__(bert_dir=bert_dir, dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        mid_linear_dims = kwargs.pop('mid_linear_dims', 128)

        self.mid_linear = nn.Sequential(
            nn.Linear(out_dims, mid_linear_dims),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        out_dims = mid_linear_dims

        self.classifier = nn.Linear(out_dims, num_tags)

        self.loss_weight = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.loss_weight.data.fill_(-0.2)

        self.crf_module = CRF(num_tags=num_tags, batch_first=True)

        init_blocks = [self.mid_linear, self.classifier]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)


    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                labels=None,
                pseudo=None):

        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

        # 常规
        seq_out = bert_outputs[0]

        seq_out = self.mid_linear(seq_out)

        emissions = self.classifier(seq_out)

        if labels is not None:
            if pseudo is not None:
                # (batch,)
                tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                    tags=labels.long(),
                                                    mask=attention_masks.byte(),
                                                    reduction='none')

                # nums of pseudo data
                pseudo_nums = pseudo.sum().item()
                total_nums = token_ids.shape[0]

                # learning parameter
                rate = torch.sigmoid(self.loss_weight)
                if pseudo_nums == 0:
                    loss_0 = tokens_loss.mean()
                    loss_1 = (rate*pseudo*tokens_loss).sum()
                else:
                    if total_nums == pseudo_nums:
                        loss_0 = 0
                    else:
                        loss_0 = ((1 - rate) * (1 - pseudo) * tokens_loss).sum() / (total_nums - pseudo_nums)
                    loss_1 = (rate*pseudo*tokens_loss).sum() / pseudo_nums

                tokens_loss = loss_0 + loss_1

            else:
                tokens_loss = -1. * self.crf_module(emissions=emissions,
                                                    tags=labels.long(),
                                                    mask=attention_masks.byte(),
                                                    reduction='mean')

            out = (tokens_loss,)

        else:
            tokens_out = self.crf_module.decode(emissions=emissions, mask=attention_masks.byte())

            out = (tokens_out, emissions)

        return out



