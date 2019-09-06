# coding: utf-8
# lstmを２層にしてDropOutを追加したもの
from ch08.attention_layer import TimeAttention
from ch07.seq2seq import Encoder, Seq2seq
from common.time_layers import *
import sys
sys.path.append('..')


class AttentionEncoder(Encoder):
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wx_1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh_1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        lstm_b_1 = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        self.lstm_1 = TimeLSTM(lstm_Wx_1, lstm_Wh_1, lstm_b_1, stateful=False)
        self.dropout = TimeDropout(dropout_ratio)
        self.dropout_1 = TimeDropout(dropout_ratio)
        self.dropout_2 = TimeDropout(dropout_ratio)

        self.params = self.embed.params + self.lstm.params + self.lstm_1.params + \
            self.dropout.params + self.dropout_1.params + self.dropout_2.params
        self.grads = self.embed.grads + self.lstm.grads + self.lstm_1.grads + \
            self.dropout.grads + self.dropout_1.grads + self.dropout_2.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        xs = self.dropout.forward(xs)
        hs = self.lstm.forward(xs)
        hs = self.dropout_1.forward(hs)
        hs = self.lstm_1.forward(hs)
        hs = self.dropout_2.forward(hs)
        self.hs = hs
        return hs

    def backward(self, dh):
        dhs = self.dropout_2.backward(dh)
        dhs = self.lstm_1.backward(dhs)
        dhs = self.dropout_1.backward(dhs)
        dout = self.lstm.backward(dhs)
        dout = self.dropout.backward(dout)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(2*H, V) / np.sqrt(2*H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        lstm_Wx_1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh_1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = (rn(4 * H)).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.lstm_1 = TimeLSTM(lstm_Wx_1, lstm_Wh_1, lstm_b1, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        self.dropout = TimeDropout(dropout_ratio)
        self.dropout_1 = TimeDropout(dropout_ratio)
        self.dropout_2 = TimeDropout(dropout_ratio)
        layers = [self.embed, self.lstm, self.lstm_1, self.attention,
                  self.affine, self.dropout, self.dropout_1, self.dropout_2]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        h = enc_hs[:, -1]
        self.lstm.set_state(h)
        self.lstm_1.set_state(h)
        self.dropout.train_flg = True
        self.dropout_1.train_flg = True
        self.dropout_2.train_flg = True
        out = self.embed.forward(xs)
        out = self.dropout.forward(out)
        dec_hs = self.lstm.forward(out)
        c = self.attention.forward(enc_hs, dec_hs)
        dec_hs = self.dropout_1.forward(dec_hs)
        rec_hs = (dec_hs + c) / 2
        rec_hs = self.lstm_1.forward(rec_hs)
        dec_hs = self.dropout_2.forward(rec_hs)
        out = np.concatenate((c, dec_hs), axis=2)
        score = self.affine.forward(out)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2
        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        dout = self.dropout_2.backward(ddec_hs0)
        dout = self.lstm_1.backward(dout)
        dout2 = dc + dout
        dout1 = self.dropout_1.backward(dout)
        denc_hs, ddec_hs1 = self.attention.backward(dout2)
        dout = dout1 + ddec_hs1
        dout = self.lstm.backward(dout)
        dout = self.dropout.backward(dout)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)
        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)
        self.lstm_1.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            out = self.dropout.forward(out)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            dec_hs = self.dropout_1.forward(dec_hs)
            rec_hs = (dec_hs + c) / 2
            rec_hs = self.lstm_1.forward(rec_hs)
            dec_hs = self.dropout_2.forward(rec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)
            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
