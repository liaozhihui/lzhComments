from gensim.models import word2vec
import jieba
import pandas as pd


text_corpus = """
我1号申请的，昨晚下班才接到回访，今天早上看审核通过了，但是还没有短信额度也查不到,
先看看有没有分期通额度??然后看自己有没有网贷??网贷用了多少??有没有逾期??我刚申请的??不到半小时就收到成功的短信...,
顺德农商_14k,2
商户问题，我也好几个月没收到里程了。上个月换了个小动物可以了。,
但平安我没有卡，因此请大家告诉我，平安野鸡在什么地方？,
标普留着就是，白麒麟提额不一定有标普快。,
我是1号提了两万，到十几号再提就出现代码了，心慌了,
征信上已经漂白了，拿去银行问了，说没问题,
卧槽，为了给你看图刚点分期变成213了，之前一直是QB9。看图,
我去年12月份，到现在都没有录入，应该是丢掉了，这效率低的,
"""
text_corpus = text_corpus.split(",")
print(text_corpus)
corpus = [jieba.lcut(line.strip()) for line in text_corpus]
model = word2vec.Word2Vec(corpus,min_count=1)
model.wv.save_word2vec_format("myvector.vector",binary=False)
