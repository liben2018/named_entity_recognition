# config LSTM training parameters
class TrainingConfig(object):
    batch_size = 4  # 64
    lr = 0.001
    epoches = 15
    print_step = 50
    # lr = 0.001
    # epoch = 30
    # print_step = 5


class LSTMConfig(object):
    # emb_size = 128  # Dimension of word-vecter 词向量的维数
    # hidden_size = 128  # Dimension of LSTM hidden states
    emb_size = 512  # Dimension of word-vecter 词向量的维数
    hidden_size = 512  # Dimension of LSTM hidden states
    # 512, 512, best 92%-f1,
    # 256 0.91-f1,
    # 128 0.90-f1,
    # 1024 0.91-f1
