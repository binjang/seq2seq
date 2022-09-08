import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder

class Seq2seq(nn.Module):
    r"""
    Inputs: inputs, targets
        - inputs: BxT (T = 인풋 길이)
        - targets: BxT (T = 타겟 길이)

    Returns:
         - outputs: BxTxD (T = 타겟 길이 또는 생성 길이 <- max decoding steps)
    """
    def __init__(self) -> None:
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(
            input_size: int,
                        embedding_size: int,
            hidden_size: int,
            num_layers: int,
            bidirectional: boll = True,
                                  rnn_type: str = "lstm",
                                                  input_dropout_p: float,
        )
        self.decoder = Decoder()

    def forward(
            self,
            inputs: torch.LongTensor,
            targets: torch.LongTensor = None, # 타겟이 None이면 인퍼런스 상황
    ):
        pass
        encoder_hidden_states, encoder_last_hidden_state = self.encoder(inputs)

        # ===========
        # TEST PASS => 이렇게 찍어두고 깃에 올리기
        # ===========


        outputs = self.decoder(targets, encoder_last_hidden_state)

        return outputs

if __name__ == "__main__":
    BATCH_SIZE = 3
    HIDDEN_DIM = 10
    SEQ_LENGTH = 12
    EMBEDDING_DIM = 10
    NUM_VOCABS = 13

    vocabs = {i : i for i in range(10)}
    vocabs['<sos>'] = 10
    vocabs['<eos>'] = 11
    vocabs['<pad>'] = 12

    test_inputs = torch.randint(low = 0, high=9, size=(BATCH_SIZE, SEQ_LENGTH))
    test_targets = torch.randint(low=0, high=9, size=(BATCH_SIZE, SEQ_LENGTH))

    model = Seq2seq(
        input_size=10,
        embedding_size=EMBEDDING_DIM,
        hidden_size=HIDDEN_DIM,
        num_encoding+layers=1,
        bidirectional=True,
    )

    outputs = model(inputs = test_inputs, targets = test_targets)
    print(outputs.size())