import jieba
import torch.nn.init
from torchtext.legacy import data
from torchtext.legacy.vocab import Vocab,Vectors
from torchtext.legacy.data import Iterator
import torchtext
"""
去停用词
"""
dirPath = "/home/lzh/PycharmProjects/lzhComments/"
def get_stop_words(Stopfile=dirPath+'data/stopwords.txt'):
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
ids = data.Field(sequential=False)

train,val = data.TabularDataset.splits(
    path=dirPath+"data",
    skip_header=True,
    train="train.tsv",
    validation='test.tsv',
    format="tsv",
    # fields=[("label",label),("text",text),],
    fields=[("label", label),("text", text), ],

)

test = data.TabularDataset(path=dirPath+"data"+"/test_public.csv",
    format="csv",
    skip_header=True,
    fields=[("id",ids),("text",text)])
print(test.examples[0].id,test.examples[0].text)

cache = dirPath+'data/.vector_cache'
vectors = Vectors(name=dirPath+'data/myvector.vector',cache=cache)
# vectors.unk_init = torch.nn.init.xavier_uniform()
text.build_vocab(train,val,test, vectors=vectors)
label.build_vocab(train, val)
ids.build_vocab(test)
batch_size=128
train_iter, val_iter = Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(val)) # 训练集设置batch_size,验证集整个集合用于测试
    )

test_iter = Iterator(test,batch_size=len(test),sort=False,sort_within_batch=False,shuffle=False)
# vocab_size = len(text.vocab)
# label_num = len(label.vocab)
# print(vocab_size)
# print(label_num)

# batch = next(iter(train_iter))
# data = batch.text
# # print(text.vocab.itos)
# # print(batch.text.shape)
# print(label.vocab.stoi)
# print(label.vocab.itos)
# print(batch.label)
# print(vectors.vectors.shape)
print(label.vocab.itos)

