import pandas as pd
import random
df = pd.read_csv("train_data_public.csv")
classification = df[["text","class"]]
indexs = [i for i in range(len(df))]
random.shuffle(indexs)
valIdx = indexs[:100]
valData = classification.iloc[valIdx]
valData.to_csv("val.csv",index=False,encoding="utf_8_sig")

classification.iloc[indexs[100:]].to_csv("train.csv",index=False,encoding="utf_8_sig")
