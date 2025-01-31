from torch.utils.data import DataLoader
from dataset import TweetDataset


class TweetLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_file(
        cls,
        file_path,
        tokenizer_name="bert-base-uncased",
        transform=None,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    ):
        dataset = TweetDataset(
            file_path=file_path,
            tokenizer_name=tokenizer_name,
            transform=transform,
        )

        return cls(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
