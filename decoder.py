import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    SUPPORTED_RNNS = {
        "rnn": nn.RNN,
        "lstm": nn.LSTM,
        "gru": nn.GRU,
    }

    def __init__(
            self,
            num_vocabs: int,
            hidden_size: int,
            num_layers: int,
            sos_id,
            rnn_type: str = "lstm",
            lstm_dropout_p: float = 0.1,
            input_dropout_p: float = 0,1,
            max_decoding_steps: int = 128,
    ) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_vocabs, hidden_size)
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn = self.SUPPORTED_RNNS[rnn_type.lower()](
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout_p,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, num_vocabs, bias=False)
        self.sos_id = sos_id
        self.max_decoding_steps = max_decoding_steps

    def forward_steps(self, input_var, hidden_state):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        outputs, hidden_state = self.rnn(embedded, hidden_state)
        # BxTxD => BxTxV
        step_output = self.fc(torch.tanh(outputs, hidden_state))
        step_output = F.log_softmax(step_output, dim=1)

        return step_output, hidden_state

    def forward(
            self,
            inputs,
            encoder_last_hidden_states,
    ):
        outputs = []
        batch_size = encoder_last_hidden_states.size(0)
        hidden_state = encoder_last_hidden_states
        step_output = inputs

        # if inference
        if inputs is None:
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)

            # di = i th decoder
            for di in range(self.max_decoding_steps):
                step_output, hidden_state = self.forward_steps(step_output, hidden_state)
                outputs.append(step_output)

        return outputs, hidden_state
