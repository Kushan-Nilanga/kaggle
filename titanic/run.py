from turtle import ontimer
from numpy import dtype, float32
import torch as t
import torch.nn as nn
import pandas as p
from sklearn import preprocessing


def clean_data(data):
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # fill missing
    cols = ['SibSp', 'Parch', 'Fare', 'Age']

    for col in cols:
        data.fillna(data[col].median(), inplace=True)

    data["Embarked"].fillna("U", inplace=True)

    return data


class LogisticRegression(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(LogisticRegression, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dims)
        )

    def forward(self, x):
        return t.sigmoid(self.sequential(x))


if __name__ == "__main__":

    # initial setup
    train = p.read_csv('./data/train.csv')
    test = p.read_csv('./data/test.csv')

    train = train.drop(['PassengerId'], axis=1)

    train = clean_data(train)
    test = clean_data(test)

    le = preprocessing.LabelEncoder()

    cols = ["Sex", "Embarked"]
    for col in cols:
        train[col] = le.fit_transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))

    print(train)
    print(test)

    # pandas to tensors
    train_x = t.tensor(train.drop(
        ['Survived'], axis=1).values, dtype=t.float32)
    train_y = t.tensor(train['Survived'], dtype=t.float32)
    train_y = train_y.reshape([train_y.shape[0], 1])

    # training
    model = LogisticRegression(7, 1)
    optimiser = t.optim.SGD(model.parameters(), lr=0.011)
    criterion = nn.BCELoss()

    # training loop
    for epoch in range(5000):
        model.train()
        y_hat = model(train_x)
        loss = criterion(y_hat, train_y)
        loss.backward()

        if epoch % 100 == 0:
            print(loss)

        optimiser.step()
        optimiser.zero_grad()

    test_x = t.tensor(
        test.drop(['PassengerId'], axis=1).values, dtype=t.float32)
    with t.no_grad():
        test_y = model(test_x)
        test_y = test_y.round().reshape([test_y.shape[0]])

    submission = p.DataFrame()
    submission["PassengerId"] = test["PassengerId"]
    submission["Survived"] = p.Series(test_y.numpy()).astype(int)


    print(submission.to_csv("submission.csv", index=False))
