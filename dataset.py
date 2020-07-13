import torch
import math
import numpy as np
import pandas as pd


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, has_label):
        self.root_dir = root_dir
        self.has_label = has_label
        self.data = pd.read_csv(root_dir + csv_file)
        self.data = self.data.sort_values(by=["length"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.data.loc[idx]
        source = np.load(self.root_dir + item["file_path"])
        if self.has_label:
            target = list(map(int, item["label"].split('_')))
            target = np.array(target)
            return source, target
        return source


def make_collate_fn(pad_idx):
    def collate_src(srcs):
        batch_src_len = max([len(src) for src in srcs])
        batch_src_len = math.ceil(batch_src_len / 8) * 8
        batch_src = []
        for src in srcs:
            src = np.pad(src, ((0, batch_src_len - len(src)), (0, 0)), 'constant', constant_values=0)
            batch_src.append(src)
        batch_src = np.stack(batch_src)
        return torch.Tensor(batch_src)

    def collate_tgt(tgts):
        batch_tgt_len = max([len(tgt) for tgt in tgts])
        batch_tgt = []
        for tgt in tgts:
            tgt = np.pad(tgt, (0, batch_tgt_len - len(tgt)), 'constant', constant_values=pad_idx)
            batch_tgt.append(tgt)
        batch_tgt = np.stack(batch_tgt)
        return torch.LongTensor(batch_tgt)

    def collate_fn(items):
        if type(items[0]) == tuple:
            srcs = collate_src([src for src, tgt in items])
            tgts = collate_tgt([tgt for src, tgt in items])
            return srcs, tgts
        else:
            return collate_src(items)
    return collate_fn
