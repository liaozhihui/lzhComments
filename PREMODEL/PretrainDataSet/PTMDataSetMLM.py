"""
基于MLM的预训练数据集，采用MacBERT的方式
"""
import re
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
from transformers import BertTokenizer
from typing import List
import logging
import pandas as pd
logger = logging.getLogger(__name__)


def get_all_tokens(sens: List[str], tokenizer: BertTokenizer):
    all_tokens = [token for sen in sens for token in tokenizer.encode(sen, add_special_tokens=False)]
    all_tokens.extend([tokenizer.sep_token, tokenizer.cls_token])
    return list(set(all_tokens))

def collect_fn(batch):
    """

    :param batch:List[data_set[i]]
    :return:
    """
    ipt_id_list = [i["input_ids"] for i in batch]
    token_type_id_list = [i["token_type_ids"] for i in batch]
    attn_mask_list = [i["attention_mask"] for i in batch]
    max_len = max([len(i) for i in ipt_id_list])
    ipt_id_list = [i + [0] * (max_len - len(i)) for i in ipt_id_list]
    token_type_id_list = [i + [0] * (max_len - len(i)) for i in token_type_id_list]
    attn_mask_list = [i + [0] * (max_len - len(i)) for i in attn_mask_list]
    input_ids = torch.tensor(ipt_id_list, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_id_list, dtype=torch.long)
    attention_mask = torch.tensor(attn_mask_list, dtype=torch.long)
    labels_list = [i["labels"] for i in batch]
    labels_list = [i + [-100] * (max_len - len(i)) for i in labels_list]
    labels = torch.tensor(labels_list, dtype=torch.long)
    return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
            "labels": labels}


class PTMDataSetMLM(Dataset):
    def __init__(self, file_path: str, num_before: int, num_after: int,
                 scene_strategy: str, tokenizer: BertTokenizer, max_len: int,
                 use_role_data: bool, surround_tarsen: bool):
        """

        :param file_path: 训练集路径
        :param num_before: 选取前几个句子,[0,]
        :param num_after: 选取后几个句子
        :param num_after: 选取后几个句子
        :param use_role_data: 是否只使用包含角色的数据
        :param scene_strategy: 场景策略，original_order：原始顺序,in_scene只选择本场景内的
        """
        # [ List[scrip_id, scene_id, sen_id, content, role, emotions]  ]
        self.data = []
        rawdata = pd.read_csv(file_path, sep="\t", encoding="utf8")
        for row in rawdata.itertuples():
            # 获取基本数据项
            data_id = str(getattr(row, "id")).strip()
            content = str(getattr(row, "content")).strip()

        self.real_data_ids = list(range(len(self.data)))
        self.data = content
        logger.info("全量数据: {}条".format(len(self.data)))
        logger.info("真正用来创建上下文的数据: {}条".format(len(self.real_data_ids)))
        self.num_before = num_before
        self.num_after = num_after
        self.scene_strategy = scene_strategy
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.surround_tarsen = surround_tarsen
        self.tokenizer.add_tokens(["[unused{}]".format(i) for i in range(1, 100)], special_tokens=True)
        logger.info("获取所有tokens")
        self.all_tokens = get_all_tokens(self.data, self.tokenizer)
        logger.info("tokens 共有{}个".format(len(self.all_tokens)))

    def __len__(self):
        return len(self.real_data_ids)

    def __getitem__(self, item):
        """
        item 为数据索引，迭代取第item条数据
        """
        # 获取目标数据
        item = self.real_data_ids[item]
        scrip_id, scene_id, sen_id, content, role, emotions = self.data[item]
        # 获取前后数据
        before_data, content, after_data = DataUtil.get_context(data=self.data, item=item, num_before=self.num_before,
                                                                num_after=self.num_after, max_len=self.max_len,
                                                                scene_strategy=self.scene_strategy)
        # 拼接整合
        # 把角色替换为unused符号
        role2token = {role: "[unused1]"} if role is not None else {}
        for i in re.finditer("[a-z]\d", "=".join(before_data) + content + "=".join(after_data)):
            t_role = i.group()
            if t_role not in role2token:
                role2token[t_role] = "[unused{}]".format(len(role2token) + 1)
        for idx in range(len(before_data)):
            before_data[idx] = DataUtil.replace_with_dict(before_data[idx], "[a-z]\d", role2token)
        for idx in range(len(after_data)):
            after_data[idx] = DataUtil.replace_with_dict(after_data[idx], "[a-z]\d", role2token)
        content = DataUtil.replace_with_dict(content, "[a-z]\d", role2token)
        # 是否针对目标句添加特殊符号
        if self.surround_tarsen:
            content = "[unused94]" + content + "[unused95]"  # 理解为 “【”和“】”
        # 拼接结束符
        for idx in range(len(before_data)): before_data[idx] += self.tokenizer.sep_token
        for idx in range(len(after_data)): after_data[idx] += self.tokenizer.sep_token
        content += self.tokenizer.sep_token
        # 拼接输入
        input_ids, token_type_ids, attention_mask = [self.tokenizer.cls_token_id], [], []
        # 前面的句子
        for sen in before_data: input_ids.extend(self.tokenizer.encode(sen, add_special_tokens=False))
        # 本句子
        input_ids.extend(self.tokenizer.encode(content, add_special_tokens=False))
        # 后面的句子
        for sen in after_data: input_ids.extend(self.tokenizer.encode(sen, add_special_tokens=False))
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)  # 定死为全0
        # 获取掩码
        input_ids, labels = DataUtil.mask_ids_macbert(ids=input_ids, tokenizer=self.tokenizer,
                                                      all_tokens=self.all_tokens, mlm_ratio=0.12)
        return {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask,
                "labels": labels}
