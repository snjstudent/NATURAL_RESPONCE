from dataset import sequence
from common.optimizer import Adam
from common.util import eval_seq2seq
from ch07.seq2seq import Seq2seq
import pickle
from ch08 import attention_seq2seq_co1 as attention_seq2seqss
from common import config
import glob
import MeCab
import numpy as np
import sys

config.GPU = False


def resize_list(wordlist, maxlen):
    wordlist.extend([-1] * maxlen)
    wordlist = wordlist[0:maxlen]


# 読み込むファイルの読み込み
word_to_id = open("word_to_id.pickle", "rb")
word_to_id = pickle.load(word_to_id)
id_to_word = open("id_to_word.pickle", "rb")
id_to_word = pickle.load(id_to_word)


# 会話データの入力
input_sentence = input()
mecab = MeCab.Tagger('-Ochasen')
input_sentence = mecab.parse(input_sentence).split("\n")
word_list = [word_to_id["RES"]]

for i in input_sentence:
    i = i.split("\t")
    i = i[0]
    if i not in word_to_id:
        i = word_to_id["UNK"]
    else:
        i = word_to_id[i]
    word_list.append(i)


resize_list(word_list, 100)
word_list = np.array([word_list])
vocab_size = len(id_to_word)
wordvec_size = 100
hidden_size = 100

model = attention_seq2seqss.AttentionSeq2seqss(
    vocab_size, wordvec_size, hidden_size)
optimizer = Adam()
model.load_params("AttentionSeq2seq_cpu.pkl")
start_id = word_list[0][0]
guess = model.generate(word_list, start_id, len(word_list[0]) - 1)
guess = ''.join([id_to_word[(int(c))] for c in guess])
print(guess)
