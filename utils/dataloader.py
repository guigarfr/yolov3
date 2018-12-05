import numpy as np
from torch.utils.data import dataloader


def collate_fn(batch):
    images = dataloader.default_collate([d['image'] for d in batch])
    annotations = [
        dataloader.default_collate(np.array(d['objects'], dtype=np.float32))
        for d in batch
    ]
    return dict(
        image=images,
        objects=annotations
    )


class YoloDataLoader(dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('collate_fn', collate_fn)
        super(YoloDataLoader, self).__init__(*args, **kwargs)
