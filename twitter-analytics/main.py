import pandas as pd
from string import punctuation

# nltk imports
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# vocab builder imports
from torchtext.data import Field
from torchtext.vocab import Vocab
from collections import Counter

# model imports
import torch as t
import torch.nn as nn


class Seq2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim=1, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(self, Seq2Vec).__init__()

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
            x_embed, x_len, batch_first=True)

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

    # remove punctuations
    x = x.apply(lambda row: row.translate(str.maketrans('', '', punctuation)))

    # tokenise
    x = x.apply(lambda row: word_tokenize(row))

    # Removal of Stopwords
    STOPWORDS = set(stopwords.words('english'))
    x = x.apply(lambda row: [word for word in row if word not in STOPWORDS])

    # stemming
    stemmer = SnowballStemmer('english')
    x = x.apply(lambda row: [stemmer.stem(word) for word in row])

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    x = x.apply(lambda row: [lemmatizer.lemmatize(word) for word in row])

    # dropping 1 or 2 letter words
    x = x.apply(lambda row: [word for word in row if len(word) > 2])

    # dropping words with more than 10 letters
    x = x.apply(lambda row: [word for word in row if len(word) < 10])

    x = x.apply(lambda row: ' '.join(row))

    x.to_csv("x.csv")
    return x


def build_vocab(data: pd.DataFrame) -> Field:
    # init the field for vocab
    vocab = Field(init_token='<sos>', eos_token='<eos>')

    # building vocabulary from data
    vocab.build_vocab(data, vectors="glove.6B.100d")
    return vocab

# embedding data from ints


def embed_data(data: pd.DataFrame, vocab: Field):
    return data.apply(lambda row: vocab.numericalize(row))


if __name__ == "__main__":
    data = pd.read_csv("./data/sentiment_tweets3.csv")
    data = preprocess(data)
    vocab = build_vocab(data)
    embeddings = embed_data(data, vocab)
    print(embeddings)
