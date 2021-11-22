import re
import os
import json
from transformers import BertTokenizer
from collections import defaultdict
import csv
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

class InputExample:

    def __init__(self,text_a,text_b=None,labels=None):
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class NERInputExample:

    def __init__(self,sid,text_a,text_b=None,labels=None):
        self.sid = sid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class BaseFeature:

    def __init__(self,token_ids,attention_masks,token_type_ids):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids



class CRFFeature(BaseFeature):

    def __init__(self,
                 token_ids,attention_masks,token_type_ids,
                 label_ids=None):
        super(CRFFeature,self).__init__(token_ids=token_ids,attention_masks=attention_masks,token_type_ids=token_type_ids)
        self.label_ids = label_ids





class NerProcessor(DataProcessor):



    def get_examples(self, data_dir, file_name):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, file_name))
        examples = []
        for (i, line) in enumerate(lines):
            sid = "%s-%s" % ("train", i)
            text_a = line[0]
            label = line[1].split(" ")
            examples.append(NERInputExample(sid=sid, text_a=text_a, label=label))
        return examples

    def get_labels(self, data_dir):
        labels = []
        with open(os.path.join(data_dir, "label.txt"), "r") as f:
            for line in f:
                labels.append(line.strip())
        return labels




def convert_example_to_feature(examples,tokenizer,max_len,label_list,):

        def convert_text_to_id(text):

            tokens = tokenizer.tokenize(text,add_special_tokens=True)
            tokens = ["[CLS]"]+tokens[:max_len-2]+["[SEP]"]
            text_len = len(text)
            tokens_id = tokenizer.convert_tokens_to_ids(tokens+["[PAD]"]*(max_len-text_len))
            attention_mask = [1]*text_len+[0]*(max_len-text_len)
            token_type_ids = [0]*max_len

            assert len(tokens_id) == len(attention_mask) == len(token_type_ids) == max_len


            return tokens,tokens_id,attention_mask,token_type_ids

        features = []

        labels_map = {label:i for i,label in enumerate(label_list)}


        for example in examples:

            tokens,token_ids,attention_masks,token_type_ids = convert_text_to_id(example.text_a)

            label_ids = [labels_map["O"]]


            for j in example.label:

                label_ids.append(labels_map[j])
            label_ids.append(labels_map["O"])

            if max_len>len(label_ids):
                label_ids = label_ids+["O"]*(max_len-len(label_ids))

            assert len(label_ids)==len(token_ids)

            feature = CRFFeature(
                # bert inputs
                token_ids=token_ids,
                attention_masks=attention_masks,
                token_type_ids=token_type_ids,
                labels=label_ids)
            features.append(feature)

        return features




processors = {
    "commentNer": NerProcessor
    }

args={

}

def load_and_cache_examples(args,tokenizer,task="commentNer",**kwargs):

    processor = processors[task]()
    label_list = processor.get_labels(args.data_dir)





