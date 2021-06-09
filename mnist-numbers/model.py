from dataset import DigitDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from collections import deque


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.model = nn.Sequential(

            # conv 1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=28, stride=1),
            nn.BatchNorm2d(8),

            # conv 2
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=2, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2),


            # linear layers
            nn.Flatten(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1),
        )

    def forward(self, x):
        return self.model(x)


def accuracy(y_hat, y):
    corr_count = 0.0
    for i in range(len(y_hat)):
        if (int(y_hat[i, 0]) == int(y[i, 0])):
            corr_count += 1.0
    return corr_count * 100 / len(y_hat)


if __name__ == '__main__':
    model = Conv()

    N_EPOCH = 20

    test = DigitDataset('test.csv', False)
    train = DigitDataset('train.csv')

    loader = DataLoader(train, 5000, shuffle=True)

    criterion = nn.MSELoss()
    optim = optim.Adam(model.parameters(), 0.05)

    model.train()
    progress = tqdm(range(N_EPOCH))

    aggr_loss = deque()
    aggr_accuracy = deque()
    j = 0
    for i in progress:
        for batch in loader:
            j += 1
            model.zero_grad()

            pred = model(batch[0].reshape(len(batch[0]), 1, 28, 28))
            y = batch[1].reshape([-1, 1])
            loss = criterion(pred, y.float())
            loss.backward()
            optim.step()

            acc = accuracy(pred, y)

            aggr_loss.append(loss)
            aggr_accuracy.append(acc)

            if(len(aggr_accuracy) > 100):
                aggr_accuracy.popleft()
                aggr_loss.popleft()

            progress.set_description(
                "r. loss: %.6f" % (sum(aggr_loss)/len(aggr_loss)) + "  r. accuracy: %.2f" % (sum(aggr_accuracy)/len(aggr_accuracy)))
