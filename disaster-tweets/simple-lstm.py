# imports
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchtext
from torchtext.data import Field, BucketIterator, Dataset, Example, LabelField

import time

import spacy
eng_tok = spacy.load('en_core_web_sm')
# tokenizer
# default tokenizer in spacy doesnt work with torchtext


def tokenizer(text):
    return [token.text for token in eng_tok.tokenizer(text)]


# dataframe dataset
# source : https://gist.github.com/lextoumbourou/8f90313cbc3598ffbabeeaa1741a11c8
# to use DataFrame as a Data source
class DataFrameDataset(Dataset):

    def __init__(self, df, fields, is_test=False, **kwargs):
        examples = []
        for i, row in df.iterrows():
            label = row.target if not is_test else None
            text = row.text
            examples.append(Example.fromlist([text, label], fields))

        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    @classmethod
    def splits(cls, fields, train_df, val_df=None, test_df=None, **kwargs):
        train_data, val_data, test_data = (None, None, None)
        data_field = fields

        if train_df is not None:
            train_data = cls(train_df.copy(), data_field, **kwargs)
        if val_df is not None:
            val_data = cls(val_df.copy(), data_field, **kwargs)
        if test_df is not None:
            test_data = cls(test_df.copy(), data_field, True, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


file_data = pd.read_csv('./train.csv')


# changing and dropping
data = pd.DataFrame()
data['text'] = file_data['text']
data['target'] = file_data['target']
# data.drop(['label (depression result)'], inplace=True, axis=1)
# data.drop(['Index'], inplace=True, axis=1)

# data info
# print("Data info")
# print("Shape :")
# print(data.info())

# normalise data
data['text'] = data['text'].apply(lambda row: row.lower())
data['text'] = data['text'].apply(
    lambda row: re.sub(r"\S*http?:\S*", '<url>', row))
data['text'] = data['text'].apply(lambda row: re.sub(r"@", "", row))
data['text'] = data['text'].apply(lambda row: re.sub(
    r"[^A-Za-z0-9()<>]", " ", row))
data['text'] = data['text'].apply(lambda row: re.sub(r"\s{2,}", " ", row))

SEED = 16
train_df, valid_df = train_test_split(
    data, random_state=SEED)


torch.manual_seed(SEED)


TEXT = Field(tokenize=tokenizer, include_lengths=True)
LABEL = LabelField(dtype=torch.float)


fields = [('text', TEXT), ('label', LABEL)]

train_ds, val_ds = DataFrameDataset.splits(
    fields, train_df=train_df, val_df=valid_df)


# build vocab
MAX_VOCAB_SIZE = 20000
TEXT.build_vocab(train_ds, max_size=MAX_VOCAB_SIZE,
                 vectors='glove.6B.200d', unk_init=torch.Tensor.zero_)
LABEL.build_vocab(train_ds)


BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator = BucketIterator.splits(
    (train_ds, val_ds),
    batch_size=BATCH_SIZE,
    sort_within_batch=True,
    device=device)


class Seq2Vec(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim, ouput_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
    ):
        super(Seq2Vec, self).__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           bidirectional=bidirectional, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        # text = [sent len, batch size]

        embedded = self.embedding(text)

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        # output = [sent len, batch size, hid dim * num directions]
        # output over padding tokens are zero tensors

        # hidden = [num layers * num directions, batch size, hid dim]
        # cell = [num layers * num directions, batch size, hid dim]

        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        # and apply dropout

        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        output = self.fc1(hidden)
        output = self.dropout(self.fc2(output))

        # hidden = [batch size, hid dim * num directions]

        return output


# Hyperparameters
num_epochs = 25
learning_rate = 0.001

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.2
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # padding


# creating instance of our LSTM_net class

model = Seq2Vec(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)

pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)
model.embedding.weight.data.copy_(pretrained_embeddings)

#  to initiaise padded to zeros
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

# print(model.embedding.weight.data)
model.to(device)


# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        text, text_lengths = batch.text

        optimizer.zero_grad()
        predictions = model(text, text_lengths).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator):

    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            acc = binary_accuracy(predictions, batch.label)

            epoch_acc += acc.item()

    return epoch_acc / len(iterator)



# train loop
t = time.time()
loss = []
acc = []
val_acc = []

for epoch in range(num_epochs):

    train_loss, train_acc = train(model, train_iterator)
    valid_acc = evaluate(model, valid_iterator)

    print(
        f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Acc: {valid_acc*100:.2f}%')

    loss.append(train_loss)
    acc.append(train_acc)
    val_acc.append(valid_acc)

print(f'time:{time.time()-t:.3f}')