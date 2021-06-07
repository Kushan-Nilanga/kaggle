import torch.optim as optim
import torch as t
import torch.nn as nn
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.data import Field
import pandas as pd
from string import punctuation

# nltk imports
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# vocab builder imports
import spacy
en = spacy.load("en_core_web_sm")


class Seq2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(Seq2Vec, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # activation to get a logistical output
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_len):

        # in -> [batch_size, sent_length]
        # out -> [batch_size, sent_len, embed_dim]
        x_embed = self.embedding(x)

        # packing
        packed_embed = nn.utils.rnn.pack_padded_sequence(
            x_embed, x_len, batch_first=True, enforce_sorted=False)

        # lstm pass
        x_out, (hidden, cell) = self.lstm(packed_embed)
        # hidden shape -> [batch_size, num_layers * num_directions, hid_dim]
        # cell shape -> [batch_size, num_layers * num_directions, hid_dim]

        # concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        # hidden shape -> [batch size, hid dim * num directions]

        x = self.fc(hidden)
        return self.sigmoid(x)


def preprocess(data: pd.DataFrame):
    # dropping index column
    data = data.drop(["Index"], axis=1)

    """
        Preprocessing Textual Data

        Lower casing
        Removal of Punctuations
        Removal of Stopwords
        Removal of Frequent words
        Removal of Rare words
        Stemming
        Lemmatization
        Removal of emojis
        Removal of emoticons
        Conversion of emoticons to words
        Conversion of emojis to words
        Removal of URLs
        Removal of HTML tags
        Chat words conversion
        Spelling correction
        """

    # converting to lowercase
    x = data['message to examine'].str.lower()

    x = x.dropna()

    # remove punctuations
    x = x.apply(lambda row: row.translate(str.maketrans('', '', punctuation)))

    # split
    x = x.apply(lambda row: word_tokenize(row))

    # Removal of Stopwords
    # DEPRECATED -> Done via spacy
    #STOPWORDS = set(stopwords.words('english'))
    #x = x.apply(lambda row: [word for word in row if word not in STOPWORDS])

    # stemming
    stemmer = SnowballStemmer('english')
    x = x.apply(lambda row: [stemmer.stem(word) for word in row])

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    x = x.apply(lambda row: [lemmatizer.lemmatize(word) for word in row])

    x = x.apply(lambda row: ' '.join(row))
    return x


def build_vocab_(data: pd.DataFrame, size: int) -> Field:
    # init the field for vocab
    vocab = Field(tokenize=en, init_token='<sos>',
                  eos_token='<eos>', batch_first=True)

    # building vocabulary from data
    vocab.build_vocab(data, max_size=size,
                      vectors='glove.6B.200d', unk_init=t.Tensor.zero_)
    return vocab

# embedding data from ints


def embed_data(data: pd.DataFrame, vocab: Field, output_shape, padding):
    x = [vocab.numericalize(row) for row in data]
    x_ten = t.zeros(output_shape, dtype=t.int32)

    for j in range(x_ten.shape[0]):
        for i in range(x_ten.shape[1]):
            if(i < len(x[j])):
                x_ten[j][i] = x[j][i]
            else:
                x_ten[j][i] = padding
    return x_ten


if __name__ == "__main__":
    data = pd.read_csv("./data/sentiment_tweets3.csv")
    x = preprocess(data)

    vocab = build_vocab_(x, 10000)

    # hyperparameters
    n_epochs = 25
    learning_r = 0.01
    vocab_dim = len(vocab.vocab)
    embed_dim = 250
    hidden_dim = 256
    out_dim = 1
    n_layers = 3
    bidirectional = True
    dropout = 0.2

    model = Seq2Vec(vocab_dim, embed_dim, hidden_dim,
                    out_dim, n_layers, bidirectional, dropout)

    criterion = nn.BCEWithLogitsLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_r)

    def accuracy(preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        # convert into float for division
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    pad_idx = vocab.vocab.stoi[vocab.pad_token]

    x_len = t.tensor([len(row) for row in x], dtype=t.int64)
    x = embed_data(x, vocab, [len(x_len), embed_dim], pad_idx)
    y = t.tensor(data['label (depression result)'])

    model.train()
    # train loops
    for epoch in range(n_epochs):
        start_idx = 0
        end_idx = 0
        steps = 100
        while(end_idx < len(x_len)):
            optimiser.zero_grad()
            y_hat = model(x[start_idx:end_idx, :], x_len[start_idx:end_idx])
            loss = criterion(y_hat, y[start_idx:end_idx])
            print(loss)
            loss.backward()
            optimizer.step()
            start_idx = end_idx
            end_idx += steps
