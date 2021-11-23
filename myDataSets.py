from torch.utils.data import Dataset
import torch



class NERDataset(Dataset):

    def __init__(self,features):
        self.nums = len(features)

        self.token_ids = [torch.tensor(feature.token_ids).long() for feature in features]
        self.attention_masks = [torch.tensor(feature.token_ids).float() for feature in features]
        self.token_type_ids = [torch.tensor(feature.token_type_ids).long() for feature in features]
        self.labels = None

        self.labels = [torch.tensor(feature.labels) for feature in features]

    def __getitem__(self, index):

        data = {
            "token_ids": self.token_ids[index],
            "attention_masks": self.attention_masks[index],
            "token_type_ids": self.token_type_ids
        }
        data["labels"] = self.labels[index]

        return data


    def __len__(self):
        return self.nums