# coding: utf-8
# Encode層,Decode層共に２層にしたもの（DropOut、重み共有を追加)
from common.base_model import BaseModel
from common.time_layers import *
import sys
sys.path.append('..')


class AttetionEncoder:
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
        return hs[:, -1, :]

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh
        dhs = self.dropout_2.backward(dhs)
        dhs = self.lstm_1.backward(dhs)
        dhs = self.dropout_1.backward(dhs)
        dout = self.lstm.backward(dhs)
        dout = self.dropout.backward(dout)
        dout = self.embed.backward(dout)
        return dout


class Decoder:
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
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.lstm_1 = TimeLSTM(lstm_Wx_1, lstm_Wh_1, lstm_b_1, stateful=True)
        self.affine = TimeAffine(embed_W.T, affine_b)  # 重み共有を行う
        self.dropout = TimeDropout(dropout_ratio)
        self.dropout_1 = TimeDropout(dropout_ratio)
        self.dropout_2 = TimeDropout(dropout_ratio)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.lstm_1, self.affine, self.dropout, self.dropout_1, self.dropout_2):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.dropout.forward(out)
        out = self.lstm.forward(out)
        out = self.dropout_1.forward(out)
        out = self.lstm_1.forward(out)
        out = self.dropout_2.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.dropout_2.backward(dout)
        dout = self.lstm_1.backward(dout)
        dout = self.dropout_1.backward(dout)
        dout = self.lstm.backward(dout)
        dout = self.dropout.backword(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            out = self.lstm_1.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
