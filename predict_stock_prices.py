from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import StockPriceDataset
from model import StockPriceModel

def train(model, train_loader, criterion, optimizer, epoch):
    losses = []
    for i, (tweet_data, open, close) in enumerate(train_loader):
        tweet_data, open, close = tweet_data.float(), open.float(), close.float()
        optimizer.zero_grad()
        close_pred = model(tweet_data, open)
        loss = criterion(close_pred, close)
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
        for tweet_data, open, close in test_loader:
            tweet_data, open, close = tweet_data.float(), open.float(), close.float()
            close_pred = model(tweet_data, open)
            loss = criterion(close_pred, close)

            preds.append(close_pred.item())
            losses.append(loss.item())

    return preds, np.mean(losses)

def main():
    # For deterministic results
    np.random.seed(42)
    torch.manual_seed(42)

    tweet_data = pd.read_csv('data/tweets_anno_vader.csv', parse_dates=['date'])
    stock_data = pd.read_csv('data/stocks.csv', parse_dates=['date'])

    num_epochs = 10
    learning_rate = 0.01

    dataset = StockPriceDataset(tweet_data, stock_data)
    train_size = int(0.7*len(dataset))
    val_size = int(0.15*len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f"{len(dataset)} instances, making train/val/test split of {train_size}/{val_size}/{test_size}")
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=None)

    model = StockPriceModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        print(f"epoch {epoch} final train loss: {train_loss}")

        _, val_loss = test(model, val_loader, criterion)
        print(f"epoch {epoch} val loss: {val_loss}")
        # TODO: save the best model?
    
    # TODO: load the best model?
    test_preds, test_loss = test(model, test_loader, criterion)
    test_opens, test_closes = zip(*[(open.item(), close.item()) for _, open, close in test_loader])
    print(f"test loss: {test_loss}")
    print("sample predictions/labels:")
    for pred, open, close in list(zip(test_preds, test_opens, test_closes))[:10]:
        print(f"predicted {pred} actual {close} (open {open})")
    print()
    #print_metrics()

if __name__ == '__main__':
    main()
