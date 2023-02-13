# coding:utf-8
import warnings
warnings.filterwarnings('ignore')           # 警告文の無視
import sys
sys.path.append('../..')  # 親ディレクトリのファイルをインポートするための設定
import time
from tqdm import tqdm
import os
import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # グラフ表示用のライブラリ
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets         # データセットの読み込み用のライブラリ
from torchtext.vocab import FastText, GloVe         # word2vecの上位互換
from network import Encoder, AttentionDecoder
from lib import lists, dicts

class CommandAnalyzer():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # パラメータ設定
        self.sen_length = 30                    # 入力文の長さ(この長さより短い場合はパディングされる)
        self.output_len = 20                    # 出力ラベルの数：19 + "_"
        self.batch_size = 746                  # バッチサイズ(同時に学習するデータの数)
        self.wordvec_size = 300                 # 辞書ベクトルの特徴の数
        self.hidden_size = 650                  # 入力文をエンコーダで変換するときの特徴の数
        self.dropout = 0.5                      # 特定の層の出力を0にする割合(過学習の抑制)
        self.max_grad = 0.25                    # 勾配の最大ノルム

        self.is_debug = True                    # デバッグ用の出力をするかのフラッグ
        self.is_predict_unk = False             # 推論時に未知語を変換するかどうかのフラッグ

        # モデルのパス
        self.test_path = '37300.txt'            # データセットのパス
        self.dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self.model_path = "example"             # 保存したモデルのパス
        self.model_num = 17                     # 保存したモデルのエポック数
        self.encoder_path = "{}/model/{}/encoder_epoch{}.pth".format(self.dir_path, self.model_path, self.model_num)
        self.decoder_path = "{}/model/{}/decoder_epoch{}.pth".format(self.dir_path, self.model_path, self.model_num)
        self.text_vocab_path = "{}/model/{}/text_vocab.pth".format(self.dir_path, self.model_path, self.model_path)
        self.label_vocab_path = "{}/model/{}/label_vocab.pth".format(self.dir_path, self.model_path)
        print("Vecotors loading ...")
        self.vectors=GloVe(dim=300)                 # GloVe(dim=300) or FastText(language="en")
        print("Loaded.")

        # 学習データの読み込み
        self.TEXT = data.Field(lower=True, batch_first=True, pad_token='<pad>', tokenize=self.tokenize, preprocessing=data.Pipeline(self.preprocessing), pad_first=True, fix_length=self.sen_length)
        self.LABEL = data.Field(batch_first=True, pad_token='<pad>')


        print("Dataset loading ...")
        (self.test_data,) = data.TabularDataset.splits(path='../dataset/data/', test=self.test_path, format='tsv', fields=[('text', self.TEXT), ('label', self.LABEL)])
        print("Loaded.")
        # print(type(self.test_data))


        #辞書ベクトルの読み込み
        self.text_vocab = torch.load(self.text_vocab_path)
        self.label_vocab = torch.load(self.label_vocab_path)
        self.vocab_size = len(self.text_vocab.itos)
        self.label_size = len(self.label_vocab.itos)
        self.TEXT.vocab = self.text_vocab
        self.LABEL.vocab = self.label_vocab

        self.test_iter = data.Iterator(
                                    (self.test_data), batch_size=self.batch_size, 
                                     device=self.device, sort=False)
        # モデルの生成
        self.encoder = Encoder(self.vocab_size, self.wordvec_size, self.hidden_size, self.dropout, self.text_vocab, is_predict_unk=self.is_predict_unk)
        self.decoder = AttentionDecoder(self.label_size, self.wordvec_size, self.hidden_size, self.dropout, self.batch_size, self.label_vocab)
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.decoder.load_state_dict(torch.load(self.decoder_path))
        self.encoder.to(self.device)                                    # GPUを使う場合
        self.decoder.to(self.device)                                    # GPUを使う場合
        self.criterion = nn.CrossEntropyLoss()                 # 損失の計算
        self.softmax = nn.Softmax(dim=1)

        if self.is_debug:

            print("test size: ", len(self.test_data))
            # print('corpus size: %d, vocabulary size: %d' % (self.corpus_size, self.vocab_size))
            print("label size: ", self.label_size)
            if len(self.test_data)%self.batch_size != 0:
                print("################## ERROR ##################")
                print(" Incorrect batch size relative to data size")
                

    # 前処理の関数(トークン化)
    def tokenize(self, s: str) -> list:
        s = s.lower()
        for p in lists.remove_words:
            s = s.replace(p, '')
        for p in dicts.replace_phrases.keys():
            s = s.replace(p, dicts.replace_phrases[p])
        s = s.replace("'s", "")
        s = re.sub(r" +", r" ", s).strip()
        return s.split()

    # 前処理の関数(リプレイスワードの処理)
    def preprocessing(self, s: str) -> str:
        if s in dicts.replace_words.keys():
            return dicts.replace_words[s]
        else:
            return s


    # Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する
    def get_max_index(self, decoder_output):
        results = []
        for h in decoder_output:
            results.append(torch.argmax(h))
        return torch.tensor(results, device=self.device).view(self.batch_size, 1)

    def test_model(self):
        with torch.no_grad():
            # モデルを評価モードへ
            self.encoder.eval()
            self.decoder.eval()
            predicts = []
            row = []

            print('Evaluating......')
            with tqdm(total = len(self.test_iter), leave=False) as bar:
                for iters in self.test_iter:
                    bar.update(1)
                    time.sleep(0.001)
                    x, l = iters.text, iters.label
                    hs, encoder_state = self.encoder(x)

                    # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、"_"のtensorをバッチサイズ分作成
                    start_char_batch = [[self.label_vocab["_"]] for _ in range(self.batch_size)]
                    decoder_input_tensor = torch.tensor(start_char_batch, device=self.device)

                    # 変数名変換
                    decoder_hidden = encoder_state

                    # バッチ毎の結果を結合するための入れ物を定義
                    batch_tmp = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)

                    for _ in range(self.output_len - 1):
                        #tqdm.write('hs : {}'.format(hs.size()))
                        decoder_output, decoder_hidden, _ = self.decoder(decoder_input_tensor, hs, decoder_hidden)
                        # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
                        decoder_input_tensor = self.get_max_index(decoder_output.squeeze())
                        #tqdm.write('dec_in_tsr : {}'.format(decoder_input_tensor.size()))
                        # バッチ毎の結果を予測順に結合
                        batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)

                    # 最初のbatch_tmpの0要素が先頭に残ってしまっているのでスライスして削除
                    predicts.append(batch_tmp[:,1:])

                    for inp, output, predict in zip(x, l, batch_tmp[:,1:]):
                        inp_ = [self.text_vocab.itos[idx] for idx in inp]
                        y = [self.label_vocab.itos[idx] for idx in output]
                        p = [self.label_vocab.itos[idx.item()] for idx in predict]
                        x_str = " ".join(inp_).replace('<pad> ', '')
                        y_str = " ".join(y[1:]).replace('<pad>', '')
                        p_str = " ".join(p).replace('<pad>', '')

                        judge = "O" if y_str == p_str else "X"
                        row.append([x_str, y_str, p_str, judge])

                
        predict_df = pd.DataFrame(row, columns=["input", "answer", "predict", "judge"])
        pd.set_option('display.max_columns', 4)
        pd.set_option('display.max_colwidth', 100)

        # 正解率を表示
        print(len(predict_df.query('judge == "O"')) / len(predict_df))
        # 0.8492
        print(predict_df.query('judge == "O"').head(10))
        # 間違えたデータを一部見てみる
        print(predict_df.query('judge == "X"').head(20))

    def plot_attention_map(self):
        # attention 可視化
        iter_ = iter(self.test_iter)
        for num in range(10000):
            iters = iter_.__next__()
            x, l = iters.text, iters.label
            with torch.no_grad():
                # モデルを評価モードへ
                self.encoder.eval()
                self.decoder.eval()
                hs, encoder_state = self.encoder(x)
                decoder_x = l[:, :-1]
                decoder_output, _, attention_weight = self.decoder(decoder_x, hs, encoder_state)
            print("-"*20)
            for i in range(2):
                with torch.no_grad():
                    for offset, xt in enumerate(x[i]):
                        #print(xt)
                        if xt != 1:
                            break
                    #print(offset)
                    df = pd.DataFrame(data=torch.transpose(attention_weight[i][offset:], 0, 1).cpu().numpy(), 
                                    columns=[self.text_vocab.itos[idx.item()] for idx in x[i][offset:]], 
                                    index=[self.label_vocab.itos[idx.item()] for idx in torch.max(decoder_output[i],1)[1]])
                    plt.figure(figsize=(12, 12)) 
                    plt.ion()
                    sns.set_context("paper", 2.5)
                    sns.heatmap(df, xticklabels = 1, yticklabels = 1, square=True, linewidths=.3, cbar_kws = dict(use_gridspec=False,location="top"))
                    plt.yticks(rotation=0)
                    plt.draw()
                    plt.pause(0.0001)
            _ = input("Enter to the next, q -> quit")
            if _ is 'q':
                break
            plt.close(2)                    # 2番目のウインドウを閉じる
            plt.close(3)                    # 3番目のウインドウを閉じる



if __name__ == "__main__":
    command_analyzer = CommandAnalyzer()
    command_analyzer.test_model()
    command_analyzer.plot_attention_map()
