from torch.utils.data import RandomSampler,DataLoader

def train(opt,model,dataset):


    train_sampler=RandomSampler(dataset)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=opt.train_batch_size,
                              sampler=train_sampler,num_workers=0)
