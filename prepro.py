import re
import os
import json
from transformers import BertTokenizer
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class InputExample:

    def __init__(self,text_a,text_b=None,labels=None):
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels



