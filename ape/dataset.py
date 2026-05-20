from random import shuffle, seed

from torch.utils.data import Dataset
from docarray import DocList

from ape.data_models import MultiModal


class CustomDataset(Dataset):
    """Dataset wrapper for DocList[MultiModal] collections."""

    def __init__(self, docs: DocList[MultiModal], transform=None):
        self.docs = docs
        self.transform = transform
        self.targets = [doc.label for doc in self.docs]

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        image = self.docs[idx].embedding
        label = self.docs[idx].label

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def split_datasets(docs: DocList[MultiModal], seed_val=123, train_percentage=0.8):
    """Shuffle and split a DocList into train/test CustomDataset objects."""
    shuffled_docs = list(docs)
    seed(seed_val)
    shuffle(shuffled_docs)
    shuffled_docs = DocList[MultiModal](shuffled_docs)

    split_idx = int(train_percentage * len(shuffled_docs))
    train_docs = shuffled_docs[:split_idx]
    test_docs = shuffled_docs[split_idx:]

    return CustomDataset(train_docs), CustomDataset(test_docs)
