import jieba
from torchtext.legacy import data
from torchtext.legacy.vocab import Vocab,Vectors
from torchtext.legacy.data import Iterator
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

cache = 'data/.vector_cache'
vectors = Vectors(name='./data/sgns.financial.word',cache=cache)
text.build_vocab(train,val, vectors=vectors)
label.build_vocab(train, val)
batch_size=128
train_iter, val_iter = Iterator.splits(
            (train, val),
            sort_key=lambda x: len(x.text),
            batch_sizes=(batch_size, len(val)) # 训练集设置batch_size,验证集整个集合用于测试
    )

# vocab_size = len(text.vocab)
# label_num = len(label.vocab)
# print(vocab_size)
# print(label_num)

batch = next(iter(train_iter))
data = batch.text
# print(text.vocab.itos)
# print(batch.text.shape)
print(label.vocab.stoi)
print(label.vocab.itos)
print(batch.label)

