from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import StockPriceDataset
from model import MAPELoss, StockPriceModel

def train(model, train_loader, criterion, optimizer, epoch):
    losses = []
    for i, (tweet_data, open, diff) in enumerate(train_loader):
        tweet_data, open, diff = tweet_data.float(), open.float(), diff.float()
        optimizer.zero_grad()
        diff_pred = model(tweet_data, open)
        loss = criterion(diff_pred, diff)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 10 == 0:
            print(f"\tepoch {epoch} inst {i} train loss: {np.mean(losses)}")

    return np.mean(losses)

def test(model, test_loader, criterion):
    preds = []
    losses = []
    with torch.no_grad():
        for tweet_data, open, diff in test_loader:
            tweet_data, open, diff = tweet_data.float(), open.float(), diff.float()
            diff_pred = model(tweet_data, open)
            loss = criterion(diff_pred, diff)

            preds.append(diff_pred.item())
            losses.append(loss.item())

    return preds, np.mean(losses)

def main():
    # For deterministic results
    np.random.seed(0)
    torch.manual_seed(0)

    tweet_data = pd.read_csv('data/tweets_anno_vader.csv', parse_dates=['date'])
    stock_data = pd.read_csv('data/stocks.csv', parse_dates=['date'])

    num_epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-3
    model_cfg = dict(
        input_size=4,
        gru_hidden_size=8,
        fc_hidden_size=4
    )
    window_size = 3
    min_tweets_per_instance = 10
    train_ratio = 0.7
    val_ratio = 0.15

    dataset = StockPriceDataset(tweet_data, stock_data, window_size=window_size, min_tweets_per_instance=min_tweets_per_instance)
    train_size = int(train_ratio*len(dataset))
    val_size = int(val_ratio*len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f"{len(dataset)} instances, making train/val/test split of {train_size}/{val_size}/{test_size}")
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=None)

    model = StockPriceModel(model_cfg)
    criterion = MAPELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        print(f"epoch {epoch} final train loss: {train_loss}")

        _, val_loss = test(model, val_loader, criterion)
        print(f"epoch {epoch} val loss: {val_loss}")
        # TODO: save the best model?
    
    # TODO: load the best model?
    test_preds, test_loss = test(model, test_loader, criterion)
    test_opens, test_diffs = zip(*[(open.item(), diff.item()) for _, open, diff in test_loader])
    print(f"test loss: {test_loss}")
    print("sample predictions/labels:")
    for pred, open, diff in list(zip(test_preds, test_opens, test_diffs))[:10]:
        print(f"predicted {pred} actual {diff} (open {open})")

if __name__ == '__main__':
    main()
