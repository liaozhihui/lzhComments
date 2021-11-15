import jieba
from torchtext.legacy import data
from torchtext.legacy.vocab import Vocab,Vectors
"""
去停用词
"""
def get_stop_words(Stopfile='data/stopwords.txt'):
    file_object = open(Stopfile, encoding='utf-8')
    stop_words = []
    for line in file_object.readlines():
        line = line[:-1]
        line = line.strip()
        stop_words.append(line)
    return stop_words

stopwords = get_stop_words()

"""
分词
"""
# def tokenizer(text):
#     return [word for word in jieba.lcut(text) if word not in stop_words]

text = data.Field(sequential=True,tokenize=jieba.lcut,stop_words=stopwords,lower=True)
label = data.Field(sequential=False)

train,val = data.TabularDataset.splits(
    path="./data",
    skip_header=True,
    train="train.csv",
    validation='val.csv',
    format="csv",
    fields=[("text",text),("label",label)]
)

print(train[1].text)