from itertools import zip_longest
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from .util import tensorized, sort_by_lengths, cal_loss, cal_lstm_crf_loss
from .config import TrainingConfig, LSTMConfig
from .bilstm import BiLSTM


class BILSTM_Model(object):
    def __init__(self, vocab_size, out_size, crf=True):
        """Func: Training and testing LSTM model.
           :param
            vocab_size:
            out_size: label number
            crf: adding crf-layer or not
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model parameters
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size

        self.crf = crf
        # The model and loss_function will be different based on crf-layer is used or not!
        if not crf:
            self.model = BiLSTM(vocab_size, self.emb_size, self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_loss
        else:
            self.model = BiLSTM_CRF(vocab_size, self.emb_size, self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_lstm_crf_loss

        # load training parameters
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # Initialize the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Initialize other parameters
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists, dev_word_lists, dev_tag_lists, word2id, tag2id):
        # sort the dataset based on the length.
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)

        B = self.batch_size
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind+B]
                batch_tags = tag_lists[ind:ind+B]

                losses += self.train_step(batch_sents, batch_tags, word2id, tag2id)

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # evaluating the trained model on dev-dataset for every epoch, save the best one!
            val_loss = self.validate(dev_word_lists, dev_tag_lists, word2id, tag2id)
            if val_loss < self._best_val_loss:
                print("Saving model of epoch_%s" % e)
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss
            print("The best loss is {:.4f} for epoch_{}".format(self._best_val_loss, e))
            print("Epoch_{}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # data
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)

        # loss computing and update weights
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # batch data
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sents, lengths = tensorized(batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # loss computing
                loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step
        return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """Return predicted results of the best model on the test dataset!"""
        # Prepare data
        word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(tensorized_sents, lengths, tag2id)

        # transfer id to tag
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # discard 'end' during decoder processing of crf.
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        """indices存有根据长度排序后的索引映射的信息
        比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        索引为2的元素映射到新的索引是1...
        下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序"""

        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """Initialize the parameters:
            vocab_size: size of vocab
            emb_size: dim of word-vector
            hidden_size: dim of hidden state
            out_size: number of labels
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)
        self.transition = nn.Parameter(torch.ones(out_size, out_size) * 1/out_size)
        # Adding a CRF layer, actually means, to learn a transition matrix with size of [out_size, out_size]
        # which initialized by uniform distribution.
        # self.transition.data.zero_()

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)
        # computing CRF scores, the size of scores is [B, L, out_size, out_size]
        # each word(char) corresponding a matrix with size of [out_size, out_size]
        # (the meaning for element located in i-th row and j-th coloum of the matrix:
        # the score for in time (t-1) tag = i, in time (t) tag = j)
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        crf_scores = emission.unsqueeze(2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)
        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """Decoding using viterbi algorithm"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]: 第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]: 第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step, :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # put results in
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)
                new_in_batch = torch.LongTensor([end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat([tags_t, new_in_batch], dim=0)  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()
            try:
                tags_t = backpointer[:batch_size_t].gather(dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1], a list of size L-1, here L-1 is because of deleted end_token,
        # 其中列表内的元素是该batch在该时刻的标记
        # resorting the order and transfer dimension to [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # return decoding results
        return tagids
