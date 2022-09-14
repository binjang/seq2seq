import torch
import torch.nn as nn
from typing import Tuple, Union

class Encoder(nn.Module):
    SUPPORTED_RNNS = {
        "rnn": nn.RNN,
        "lstm": nn.LSTM,
        "gru": nn.GRU,
    }

    def __init__(
            self,
            input_size: int,
            embedding_size: int,
            hidden_size: int,
            num_layers: int,
            bidirectional: boll = True,
            rnn_type: str = "lstm",
            input_dropout_p: float,
    ) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.input_dropout =  nn.Dropout(p = input_dropout_p)
        self.rnn = self.SUPPORTED_RNNS[rnn_type.lower()](
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=input_dropout_p,
            bidirectional=bidirectional
            )

    def forward(self, inputs) -> Tuple(torch.FloatTensor, Union(torch.FloatTensor, Tuple(torch.FloatTensor, torch.FloatTensor))):
        embedded = self.embedding(inputs)
        embedded = self.input_dropout(embedded)

        hidden_states, last_hidden_state = self.rnn(embedded)

        return hidden_states, last_hidden_state
