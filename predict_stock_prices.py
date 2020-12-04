from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import StockPriceDataset
from model import StockPriceRegressor

def train(model, train_loader, criterion, optimizer, epoch, clf):
    losses = []
    for i, (tweet_data, open, label) in enumerate(train_loader):
        tweet_data, open, label = tweet_data.float(), open.float(), label.float()
        if clf:
            label = (label > 0).float()
        optimizer.zero_grad()
        pred = model(tweet_data, open)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if i % 50 == 0:
            print(f"\tepoch {epoch} inst {i} train loss: {np.mean(losses)}")
            print(pred.item(), label.item())

    return np.mean(losses)

def test(model, test_loader, criterion, clf):
    preds = []
    losses = []
    with torch.no_grad():
        for tweet_data, open, label in test_loader:
            tweet_data, open, label = tweet_data.float(), open.float(), label.float()
            if clf:
                label = (label > 0).float()
            pred = model(tweet_data, open)
            loss = criterion(pred, label)

            preds.append(pred.item())
            losses.append(loss.item())

    return preds, np.mean(losses)

def main():
    # For deterministic results
    np.random.seed(0)
    torch.manual_seed(0)

    tweet_data = pd.read_csv('data/tweets_anno_vader.csv', parse_dates=['date'])
    stock_data = pd.read_csv('data/stocks.csv', parse_dates=['date'])

    num_epochs = 10
    learning_rate = 1e-3
    weight_decay = 1e-3
    model_cfg = dict(
        input_size=4,
        hidden_size=16
    )
    window_size = 3
    min_tweets_per_instance = 1
    train_ratio = 0.7
    val_ratio = 0.15

    dataset = StockPriceDataset(tweet_data, stock_data, window_size=window_size, min_tweets_per_instance=min_tweets_per_instance)
    train_size = int(train_ratio*len(dataset))
    val_size = int(val_ratio*len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(f"{len(dataset)} instances, making train/val/test split of {train_size}/{val_size}/{test_size}")
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None)
    test_loader = DataLoader(test_dataset, batch_size=None)

    '''
    # Classification
    print("classification")
    model = StockPriceClassifier(model_cfg)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, clf=True)
        print(f"epoch {epoch} final train loss: {train_loss}")

        _, val_loss = test(model, val_loader, criterion, clf=True)
        print(f"epoch {epoch} val loss: {val_loss}")
        # TODO: save the best model?
    
    # TODO: load the best model?
    test_preds, test_loss = test(model, test_loader, criterion, clf=True)
    print(f"test loss: {test_loss}")
    test_preds = [(p > 0.5) for p in test_preds]
    test_labels = [(label.item() > 0) for _, _, label in test_loader]
    print(metrics.classification_report(test_labels, test_preds))

    print()
    '''

    # Regression
    print("regression")
    model = StockPriceRegressor(model_cfg)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, epoch, clf=False)
        print(f"epoch {epoch} final train loss: {train_loss}")

        _, val_loss = test(model, val_loader, criterion, clf=False)
        print(f"epoch {epoch} val loss: {val_loss}")
        # TODO: save the best model?
    
    # TODO: load the best model?
    test_preds, test_loss = test(model, test_loader, criterion, clf=False)
    print(f"test loss: {test_loss}")
    test_opens, test_labels = zip(*[(open.item(), label.item()) for _, open, label in test_loader])
    print("sample predictions/labels:")
    for pred, open, label in list(zip(test_preds, test_opens, test_labels))[:10]:
        print(f"predicted {pred} actual {label} (open {open})")

if __name__ == '__main__':
    main()
