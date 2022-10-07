import torch
from dataset import CocoDataset

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    coco = CocoDataset(
        root=root,
        json=json,
        vocab=vocab,
        transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=coco,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader