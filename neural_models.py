import torch
from torch import nn


class Model(torch.nn):

    def __init__(self, char_to_row, embedding_dim):

        self.embedding = nn.Embedding(len(char_to_row), embedding_dim=embedding_dim)
        self.char_lang_model = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)