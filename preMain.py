from prepro import processors,convert_example_to_feature
from myDataSets import NERDataset
from models.bert_models import build_model


def train(opt,model,dataset):


    pass


def train_base(opt,examples,processor):
    labels = processor.get_labels(opt.data_dir)
    train_features = convert_example_to_feature(examples,opt.bert_dir,opt.max_len,label_list=labels)
    train_dataset = NERDataset(train_features)
    model = build_model("crf",bert_dir=opt.bert_dir,num_tags=len(labels))




def training(opt):

    processor = processors.get("commentNer")()
    train_examples = processor.get_examples("./data","train.csv")
    train_base(opt,train_examples,processor)
