import sys
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
import pickle
from ch08 import attention_seq2seq_co1 as attention_seq2seqss
from common import config
import glob

config.GPU = True
is_reverse = False


def save_grads(grads):
    with open("/Users/rem_0202/Downloads/Natural_Responce/gradsofvec.pickle", "wb") as f:
        pickle.dump(grads, f)
        print("途中経過を保存しました")
        print(grads)


def resize_list(wordlist, maxlen, a):
    for i in range(len(wordlist)):
        wordlist[i].extend([a] * maxlen)
        wordlist[i] = wordlist[i][0:maxlen]


def resize_list_a(wordlist, maxlen, a):
    wordlist.extend([a] * maxlen)
    wordlist = wordlist[0:maxlen]


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " %
                  (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


# データセットの読み込み
x_train = []
y_train = []
x_test = []
y_test = []


b = open("/Users/rem_0202/Downloads/Natural_Responce/word_to_id.pickle", "rb")
char_to_id = pickle.load(b)
b = open("/Users/rem_0202/Downloads/Natural_Responce/id_to_word.pickle", "rb")
id_to_char = pickle.load(b)


"""
#b = open("/Users/rem_0202/Downloads/Natural_Responce/encode.pickle", "rb")
x_train = pickle_load(
    "/Users/rem_0202/Downloads/Natural_Responce/encode.pickle")
#b = open("/Users/rem_0202/Downloads/Natural_Responce/decode.pickle", "rb")
y_train = pickle_load(
    "/Users/rem_0202/Downloads/Natural_Responce/decode.pickle")



# テストデータと訓練データに分割
u = int(len(x_train) * 0.05)
x_test = x_train[-u:]
x_train = x_train[:-u]


# テストデータと訓練データに分割
u = int(len(y_train) * 0.05)
y_test = y_train[-u:]
y_train = y_train[: - u]
"""


# ハイパーパラメーターの設定
vocab_size = len(char_to_id)
wordvec_size = 100
hidden_size = 100
batch_size = 128
max_epoch = 100
max_grad = 5.0


# モデル/オプティマイザ/トレーナーの生成
model = attention_seq2seqss.AttentionSeq2seqss(
    vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
file_name_encode = "encode_"
file_name_decode = "decode_"
x_train_list = []
y_train_list = []

for i in range(101):
    file_name_encode_kari = "/Users/rem_0202/Downloads/Natural_Responce/encode_list/" + \
        file_name_encode + str(i)
    file_name_decode_kari = "/Users/rem_0202/Downloads/Natural_Responce/decode_list/" + \
        file_name_decode + str(i)
    x_train_list.extend(glob.glob(file_name_encode_kari))
    y_train_list.extend(glob.glob(file_name_decode_kari))

model.load_params(
    "/Users/rem_0202/Downloads/Natural_Responce/AttentionSeq2seqss.pkl")
# テストデータに対する評価
x_test = open(x_train_list[-1], "rb")
x_test = pickle.load(x_test)
y_test = open(y_train_list[-1], "rb")
y_test = pickle.load(y_test)
x_test = x_test[0:30]
y_test = y_test[0:30]


for epoch in range(max_epoch):
    # 分割したファイルを分けて開く
    for i in range(3, len(x_train_list)):
        x_train = open(x_train_list[i], "rb")
        x_train = pickle.load(x_train)
        y_train = open(y_train_list[i], "rb")
        y_train = pickle.load(y_train)
        for u in range(301):
            trainer.fit(x_train[u * 100:(u + 1) * 100], y_train[u * 100:(u + 1) * 100], max_epoch=1, batch_size=batch_size, max_grad=max_grad)
        if i%100==0:
            model.save_params()
        print(int(i / len(x_train_list)), "%", "completed...")
        if i % 2 == 0:
            # 念の為、訓練データのメモリ解放を行う
            x_train = ""
            y_train = ""
            # テストデータに対する評価
            x_test = open(x_train_list[-1], "rb")
            x_test = pickle.load(x_test)
            y_test = open(y_train_list[-1], "rb")
            y_test = pickle.load(y_test)
            x_test = x_test[0:30]
            y_test = y_test[0:30]
            correct_num = 0
            for u in range(len(x_test)):
                question, correct = x_test[[u]], y_test[[u]]
                verbose = u < 10
                correct_num += eval_seq2seq(model, question,
                                            correct, id_to_char, verbose, is_reverse)
                acc = float(correct_num) / len(x_test)
                acc_list.append(acc)
                print('val acc %.3f%%' % (acc * 100))

    # 念の為、訓練データのメモリ解放を行う
    x_train = ""
    y_train = ""
    # テストデータに対する評価
    x_test = open(x_train_list[-1], "rb")
    x_test = pickle_load(x_test)
    y_test = open(y_train_list[-1], "rb")
    y_test = pickle.load(y_test)
    correct_num = 0
    for u in range(len(x_test)):
        question, correct = x_test[[u]], y_test[[u]]
        verbose = u < 10
        correct_num += eval_seq2seq(model, question,
                                    correct, id_to_char, verbose, is_reverse)
    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))


"""
for epoch in range(max_epoch):
    trainer.fit(x_train, y_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)
    model.save_params()
    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], y_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

# グラフの描画
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.show()
"""
