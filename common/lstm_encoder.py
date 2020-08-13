import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """
    This module encodes a sequence into a single vector using an LSTM.
    """

    def __init__(self, in_features, hidden_size: int = 200,
                 batch_first: bool = True,
                 bidirectional: bool = True):
        """
        :param in_features:
        :param hidden_size:
        :param batch_first:
        :param bidirectional:
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=batch_first,
                            bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def forward(self, x1, mask, lengths):
        """
        Encode sentence x
        :param x: sequence of word embeddings, shape [B, T, E]
        :param mask: byte mask that is 0 for invalid positions, shape [B, T]
        :param lengths: the lengths of each input sequence [B]
        :return:
        """
        # print('use LSTM !!!!!!!!!!!!!')

        # packed_sequence = pack_padded_sequence(x, lengths, batch_first=True)
        # outputs, (hx, cx) = self.lstm(packed_sequence)
        # outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # print(outputs.size(), hx.size())

        num_directions = 2 if self.bidirectional else 1
        max_batch_size = x1.size(0) if self.batch_first else input.size(1)
        zeros = torch.zeros( num_directions,
                            max_batch_size, self.hidden_size,
                            dtype=x1.dtype, device=x1.device)
        hx = (zeros, zeros)
        x1 = x1.transpose(0,1)
        packed_input = x1
        if mask is not None:  # seq, batch
            mask = torch.transpose(mask, 0,1).float()
            packed_out = []
            outputs = []
            for x, m in zip(packed_input, mask):
                m = m.unsqueeze(-1)
                x_post, (h1,c1) = self.lstm(x.unsqueeze(1), hx)
                # print(mask[ind,:].size(), hidden1[0].size())
                # x_post = m * x_post[:,0,:] + (1-m) * x
                h1 = m * h1 + (1-m) * hx[0]  # (num_layers * num_directions, batch, hidden_size)
                c1 = m * c1 + (1-m) * hx[1]
                hx = (h1,c1)
                packed_out.append(h1)
                outputs.append(x_post[:,0,:])
            # hx = torch.stack(packed_out) # (seq, num_layers * num_directions, batch, hidden_size)
            hx = packed_out[-1]
            outputs = torch.stack(outputs)  #  (seq_len, batch, num_directions * hidden_size)

            # print(outputs.size(), hx.size())
        # classify from concatenation of final states
        if self.lstm.bidirectional:
            final = torch.cat([hx[-2], hx[-1]], dim=-1)
        else:  # classify from final state
            final = hx[-1]

        return outputs.transpose(0,1), final
