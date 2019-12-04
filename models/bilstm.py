import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """Initializing parameters:
         vocab_size: size fo vocab
           emb_size: dimension of word-vector
        hidden_size: dimension of hidden states
           out_size: number of labels

        CLASS torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0,
                                scale_grad_by_freq=False, sparse=False, _weight=None)
        Parameters:
        num_embeddings (int) – size of the dictionary of embeddings
        embedding_dim (int) – the size of each embedding vector
        padding_idx (int, optional) – If given, pads the output with the embedding vector at padding_idx
                                      (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional) – If given, each embedding vector with norm larger than max_norm is renormalized
                                     to have norm max_norm.
        norm_type (float, optional) – The p of the p-norm to compute for the max_norm option. Default 2.
        scale_grad_by_freq (boolean, optional) – If given, this will scale gradients by the inverse of frequency of
                                                 the words in the mini-batch. Default False.
        sparse (bool, optional) – If True, gradient w.r.t. weight matrix will be a sparse tensor.
                                  See Notes for more details regarding sparse gradients.

        Pytorch, nn.LSTM: https://pytorch.org/docs/stable/nn.html
        Parameters
        input_size – The number of expected features in the input x
        hidden_size – The number of features in the hidden state h
        num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together
                     to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing
                     the final results. Default: 1
        bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                  with dropout probability equal to dropout. Default: 0
        bidirectional – If True, becomes a bidirectional LSTM. Default: False

        CLASStorch.nn.Linear(in_features, out_features, bias=True), Parameters:
        in_features – size of each input sample
        out_features – size of each output sample
        bias – If set to False, the layer will not learn an additive bias. Default: True
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        # self.bilstm = nn.LSTM(emb_size, hidden_size, batch_first=True, bidirectional=True)
        self.bilstm = nn.LSTM(input_size=emb_size,
                              hidden_size=hidden_size,
                              num_layers=2,
                              bias=True,
                              batch_first=True,
                              dropout=0.8,
                              bidirectional=True)
        self.lin = nn.Linear(in_features=2*hidden_size,
                             out_features=out_size,
                             bias=True)

    def forward(self, sents_tensor, lengths):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        #
        """ avoid warning:
        /opt/conda/conda-bld/pytorch_1565272279342/work/aten/src/ATen/native/cudnn/RNN.cpp:1236: 
        UserWarning: RNN module weights are not part of single contiguous chunk of memory. 
        This means they need to be compacted at every call, possibly greatly increasing memory usage. 
        To compact weights again call flatten_parameters().
        """
        self.bilstm.flatten_parameters()
        #
        rnn_out, _ = self.bilstm(packed)  # rnn_out: [B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        scores = self.lin(rnn_out)  # [B, L, out_size]
        return scores

    def test(self, sents_tensor, lengths, _):
        """The 3rd parameter is not used here, the reason we adding the 3rd parameter here is to keep same interface
        with BiLSTM_CRF model!"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)  # get the final indices for classification results for each tag!
        """
        >>> a = torch.randn(4, 4)
        >>> a
        tensor([[-1.2360, -0.2942, -0.1222,  0.8475],
                [ 1.1949, -1.1127, -2.2379, -0.6702],
                [ 1.5717, -0.9207,  0.1297, -1.8768],
                [-0.6172,  1.0036, -0.6060, -0.2432]])
        >>> torch.max(a, 1)
        torch.return_types.max(values=tensor([0.8475, 1.1949, 1.5717, 1.0036]), indices=tensor([3, 0, 0, 1]))
        """

        return batch_tagids
