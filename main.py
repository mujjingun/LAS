import argparse
import pickle
import torch
import math
import model
import tqdm
import dataset
import numpy as np
import os
import jiwer


def main(args):
    # select device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load mapping
    with open(args.path + 'mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    pad_idx = max(mapping.values()) + 1
    vocab_size = pad_idx + 1

    # load dataset
    data = dataset.SpeechDataset('train-clean-100.csv', args.path, True)

    # train validation split
    train_size = math.ceil(len(data) * 0.9)
    val_size = len(data) - train_size
    train, val = torch.utils.data.random_split(data, [train_size, val_size])

    # make dataloaders
    collate_fn = dataset.make_collate_fn(pad_idx)
    train = torch.utils.data.DataLoader(
        train,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4
    )
    val = torch.utils.data.DataLoader(
        val,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4
    )
    test = torch.utils.data.DataLoader(
        dataset.SpeechDataset('test-clean.csv', args.path, False),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4
    )

    # construct model
    las_model = model.LAS(device, vocab_size, pad_idx)

    # load model
    if os.path.exists(args.file_name):
        print("Loading model from file ", args.file_name)
        las_model.load(args.file_name)
        print("Loaded.")

    if not args.test:
        for epoch in range(args.epochs):
            print("Epoch ", epoch)

            train_losses = []
            val_losses = []

            pbar = tqdm.tqdm(train)
            for source, target in pbar:
                source, target = source.to(device), target.to(device)
                loss = las_model.train_step(source, target)
                train_losses.append(loss)
                pbar.set_description("Loss = {:.6f}".format(loss))
            print("Train loss ", np.mean(train_losses))

            for source, target in val:
                source, target = source.to(device), target.to(device)
                with torch.no_grad():
                    loss = las_model.loss(source, target).item()
                val_losses.append(loss)
            print("Val loss ", np.mean(val_losses))

            # save model
            print("Saving to ", args.file_name)
            las_model.save(args.file_name)
            print("Saved.")
    else:
        for source, target in tqdm.tqdm(test):
            # TODO: test
            # TODO: evaluate WER
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LAS')
    parser.add_argument(
        '--path',
        type=str,
        default='../libri_fbank40_char30/'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100
    )
    parser.add_argument(
        '--test',
        action='store_true'
    )
    parser.add_argument(
        '--file_name',
        type=str,
        default="model/checkpoint.t7"
    )
    main(parser.parse_args())
