# coding:utf-8
import warnings
warnings.filterwarnings('ignore')           # 警告文の無視
import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
import time
from tqdm import tqdm
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
from torchtext.vocab import FastText, GloVe, vocab        # word2vecの上位互換
from collections import Counter
from network import Encoder, AttentionDecoder
from lib import lists, dicts
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.transforms as T
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from sklearn.model_selection import train_test_split

class CommandAnalyzer():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # パラメータ設定
        self.sen_length = 30                    # 入力文の長さ(この長さより短い場合はパディングされる)
        self.output_len = 20                    # 出力ラベルの数：19 + "_"
        self.max_epoch = 100                    # エポック数(学習回数)の最大値
        self.batch_size = 987                   # バッチサイズ(同時に学習するデータの数)
        self.wordvec_size = 300                 # 辞書ベクトルの特徴の数
        self.hidden_size = 650                  # 入力文をエンコーダで変換するときの特徴の数
        self.dropout = 0.5                      # 特定の層の出力を0にする割合(過学習の抑制)
        self.learning_rate = 0.001              # 学習率(どれくらいモデルを更新するか)の割合
        self.max_grad = 0.25                    # 勾配の最大ノルム
        self.eval_interval = 20                 # 検証する間隔(インターバル)
        self.early_stoping = 10                 # 学習が連続して向上しなかった際に中断するためのしきい値

        self.is_debug = True                    # デバッグ用の出力をするかのフラッグ
        self.is_save_vec = True                 # 辞書ベクトルを保存するかどうかのフラッグ
        self.is_save_model = True               # 学習モデルを保存するかどうかのフラッグ
        self.is_test_model = True               # モデルのテストを行うかどうかのフラッグ
        self.is_predict_unk = False             # 推論時に未知語を変換するかどうかのフラッグ

        self.train_path = 'train_98700.txt'           # データセットのパス
        self.test_path = None                   # 学習データと別のデータセットでテストを行う際のデータセットのパス
        self.model_path = "example"             # モデルを保存する際のパス
        self.text_vocab_path = "text_vocab_01.pth"
        self.label_vocab_path = "label_vocab_01.pth"
        self.vectors=GloVe(name='840B', dim=300)        # GloVe(name='840B', dim=300) or FastText(language="en")
        self.label_tokenizer = get_tokenizer(tokenizer = None)

        # 学習データの読み込み
        # self.TEXT = data.Field(lower=True, batch_first=True, pad_token='<pad>', tokenize=self.tokenize, preprocessing=data.Pipeline(self.preprocessing), pad_first=True, fix_length=self.sen_length)
        # self.LABEL = data.Field(batch_first=True, pad_token='<pad>')
        

        df = pd.read_table('../dataset/data/' + self.train_path)
        df['text'] = df['text'].map(lambda x: self.tokenize(self.preprocessing(x)))
        df['label'] = df['label'].map(lambda x: self.label_tokenizer(self.preprocessing(x)))
        
        #学習用、検証用、テスト用に分割
        self.train_text_data, self.val_text_data, self.train_label_data, self.val_label_data = train_test_split(df['text'], df['label'], train_size= 0.7)
        self.val_text_data, self.test_text_data, self.val_label_data, self.test_label_data = train_test_split(self.val_text_data, self.val_label_data, test_size= 2/3)
        self.train_data = pd.DataFrame({'text':self.train_text_data,'lavel':self.train_label_data})
        self.val_data = pd.DataFrame({'text':self.val_text_data,'lavel':self.val_label_data})
        self.test_data = pd.DataFrame({'text':self.test_text_data,'lavel':self.test_label_data})

        #traindataからテキスト辞書作成
        self.text_vocab = build_vocab_from_iterator(self.train_text_data, specials=('<unk>', '<pad>'))#, vectors=self.vectors)
        self.text_vocab.set_default_index(self.text_vocab['<unk>'])
        self.text_vectors = self.vectors.get_vecs_by_tokens(self.text_vocab.get_itos())
        # print(self.text_vocab.get_itos())
        # print(self.text_vectors)
        # ラベル辞書作成
        # df['label'] = df['label'].astype(str)  # ラベルが数値のため文字型に変換。文字型で指定していた場合はこの処理は不要

        self.label_vocab = build_vocab_from_iterator(self.train_label_data, specials=('<unk>', '<pad>'))
        self.label_vocab.set_default_index(self.label_vocab['<unk>'])
        self.label_vectors = self.vectors.get_vecs_by_tokens(self.label_vocab.get_itos())
        
        self.text_transform = T.Sequential(
            T.VocabTransform(self.text_vocab),
            T.ToTensor(padding_value=self.text_vocab['<pad>'])
        )
        self.label_transform = T.Sequential(
            T.VocabTransform(self.label_vocab),
            T.ToTensor(padding_value=self.label_vocab['<pad>'])
        )
        # print(self.label_vectors)
        # DataLoader設定
        self.train_data_loader = DataLoader(self.train_data.values, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)
        self.val_data_loader = DataLoader(self.val_data.values, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)        
        self.test_data_loader = DataLoader(self.test_data.values, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_batch)        
        # for i, (texts, labels) in enumerate(data_loader):
        #     print(i)
        #     for text, label in zip(texts, labels):
        #         print(type(text), type(label))
        if self.is_save_vec:
            torch.save(self.text_vocab, "../model/{}/{}".format(self.model_path, self.text_vocab_path))
            torch.save(self.label_vocab, "../model/{}/{}".format(self.model_path, self.label_vocab_path))
        self.label_size = len(self.label_vocab.get_itos())
        self.vocab_size = len(self.text_vocab.get_itos())
        # print(self.text_vocab.get_stoi()['what'])
        # print(len(self.text_vectors[self.text_vocab.get_stoi()['what']]))

        # モデルの生成
        self.encoder = Encoder(self.vocab_size, self.wordvec_size, self.hidden_size, self.dropout, self.text_vocab, vocab_vectors=self.text_vectors, is_predict_unk=self.is_predict_unk)
        self.decoder = AttentionDecoder(self.label_size, self.wordvec_size, self.hidden_size, self.dropout, self.batch_size, self.label_vocab)
        self.encoder.to(self.device)                                    # GPUを使う場合
        self.decoder.to(self.device)                                    # GPUを使う場合
        self.criterion = nn.CrossEntropyLoss()                 # 損失の計算
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)      # パラメータの更新
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=self.learning_rate)      # パラメータの更新
        self.softmax = nn.Softmax(dim=1)

        if self.is_debug:
            print("train size: ", len(self.train_data))
            print("val size: ", len(self.val_data))
            print("test size: ", len(self.test_data))
            print('corpus size: %d, vocabulary size: %d' % (len(self.train_data_loader), self.vocab_size))
            print("label size: ", self.label_size)
            if len(self.train_text_data)%self.batch_size != 0 or len(self.val_text_data)%self.batch_size != 0 or len(self.test_text_data)%self.batch_size != 0:
                print("################## ERROR ##################")
                print(" Incorrect batch size relative to data size")
                
    # 前処理の関数(トークン化)
    def preprocessing(self, s: str) -> str:
        if s in dicts.replace_words.keys():
            return dicts.replace_words[s]
        else:
            return s

    def tokenize(self, s: str) -> list:
        s = s.lower()
        for p in lists.remove_words:
            s = s.replace(p, '')
        for p in dicts.replace_phrases.keys():
            s = s.replace(p, dicts.replace_phrases[p])
        s = s.replace("'s", "")
        s = re.sub(r" +", r" ", s).strip()
        for p in dicts.replace_words.keys():
            s = s.replace(p, dicts.replace_words[p])
        return s.split()
    # ミニバッチ時のデータ変換関数

    def collate_batch(self, batch):
        texts = self.text_transform([text for (text, label) in batch])
        labels = self.label_transform([label for (text, label) in batch])
        return texts, labels

    # # 前処理の関数(リプレイスワードの処理)
    # def preprocessing(self, s: str) -> str:
    #     if s in dicts.replace_words.keys():
    #         return dicts.replace_words[s]
    #     else:
    #         return s

    # Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する
    def get_max_index(self, decoder_output):
        results = []
        for h in decoder_output:
            results.append(torch.argmax(h))
        return torch.tensor(results, device=self.device).view(self.batch_size, 1)

    def evaluate(self):
        loss_sum = 0
        loss_count = 0
        self.encoder.eval()
        self.decoder.eval()
        with tqdm(total = len(self.val_data_loader), leave=False) as bar:
            for i, iters in enumerate(self.val_data_loader):
                bar.update(1)
                time.sleep(0.001)
                # 入力データと教師データを取り出す
                x, l = iters[0], iters[1]
                x = x.to(self.device)
                l = l.to(self.device)
                # 順伝播
                hs, encoder_state = self.encoder(x)
                decoder_x = l[:, :-1]
                target = l[:, 1:]
                loss = 0
                # 損失計算
                decoder_output, _, attention_weight = self.decoder(decoder_x, hs, encoder_state)
                for j in range(decoder_output.size()[1]):
                    # バッチ毎にまとめてloss計算
                    loss += self.criterion(decoder_output[:, j, :], target[:, j])
                    loss_count += 1
                loss_sum += loss.item()
        loss_mean = (loss_sum / loss_count)
        return loss_mean


    def save_model(self, epoch):
        print('Saving Model ...epoch:{}'.format(epoch+1))
        encoder_path = "../model/{}/encoder_epoch{}.pth".format(self.model_path, epoch+1)
        decoder_path = "../model/{}/decoder_epoch{}.pth".format(self.model_path, epoch+1)
        # torch.save(self.encoder.state_dict(), encoder_path)
        # torch.save(self.decoder.state_dict(), decoder_path)
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def plot_loss(self, ylim=None, loss_list=[], val_loss_list=[]):
        plt.clf()
        x = np.arange(1, len(loss_list) + 1)
        vx = np.arange(1, len(val_loss_list) + 1)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, loss_list, label='学習', linestyle='solid')
        plt.plot(vx, val_loss_list, label='検証', linestyle='dashed')
        plt.xlabel('学習回数', fontsize=18)
        plt.ylabel('損失', fontsize=18)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=18)
        plt.draw()
        plt.pause(0.001)


    def train_model(self):
        # 損失の合計
        loss_sum = 0
        loss_count = 0
        stop_count = 0
        # lossのリスト
        loss_list = []
        val_loss_list = []
        # 最良のloss (初期値は無限大)
        best_loss = float('inf')
        if self.is_debug:
            for i, iters in enumerate(self.test_data_loader):
                x, l = iters[0], iters[1] # x = text, l = label
                if x.size(0) != self.batch_size or x.size(1) != self.sen_length:
                    print('x', i, x.size(), x)
                if l.size(0) != self.batch_size or l.size(1) != self.output_len:
                    print('l', i, l.size(), l)
            print(self.encoder)
            print(self.decoder)

        start_time = time.time()
        for epoch in range(self.max_epoch):
            # モデルを訓練モードに設定
            self.encoder.train()
            self.decoder.train()
            with tqdm(total = len(self.train_data_loader), leave=False) as bar:
                for i, iters in enumerate(self.train_data_loader):
                    bar.update(1)
                    time.sleep(0.001)
                    # 入力データと教師データを取り出す
                    x, l = iters[0], iters[1]
                    #print('x : ', x.size())
                    #print('l : ', l.size())
                    x = x.to(self.device)
                    l = l.to(self.device)

                    # 勾配をゼロにリセット
                    self.encoder_optimizer.zero_grad()
                    self.decoder_optimizer.zero_grad()
                    
                    # 順伝播
                    
                    hs, encoder_state = self.encoder(x)
                    decoder_x = l[:, :-1]
                    target = l[:, 1:]
                    decoder_output, _, attention_weight = self.decoder(decoder_x, hs, encoder_state)

                    # 損失計算
                    loss = 0
                    for j in range(decoder_output.size()[1]):
                        # バッチ毎にまとめてloss計算
                        # 生成する文字は2文字なので、2回ループ
                        loss += self.criterion(decoder_output[:, j, :], target[:, j])
                        loss_count += 1
                    
                    loss_sum += loss.item()

                    # 逆伝播
                    loss.backward(retain_graph=True)

                    # 勾配クリッピング
                    nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad)
                    nn.utils.clip_grad_norm_(self.decoder.parameters(), self.max_grad)

                    # 重み更新
                    self.encoder_optimizer.step()
                    self.decoder_optimizer.step()

                # lossを計算
                loss_mean = (loss_sum / loss_count)
                loss_list.append(loss_mean)
                loss_sum, loss_count = 0, 0

                # Validation
                val_loss_mean = self.evaluate()
                val_loss_list.append(val_loss_mean)
                tqdm.write('| epoch {} | loss {:.9f} |'.format(epoch+1, val_loss_mean))

                if best_loss > val_loss_mean:
                    best_loss = val_loss_mean
                    stop_count = 0
                    # モデルの保存
                    if self.is_save_model:
                        self.save_model(epoch)
                    best_encoder = self.encoder
                    best_decoder = self.decoder
                else:
                    self.learning_rate /= 4.0
                    stop_count += 1
                    if stop_count == self.early_stoping:
                        print("\n----Early stopping----")
                        break
                    for group in self.encoder_optimizer.param_groups:
                        group['lr'] = self.learning_rate
                
                self.plot_loss(None, loss_list, val_loss_list)

        end_time = time.time()
        print("learning time : {}".format(end_time-start_time))
        self.encoder = best_encoder
        self.decoder = best_decoder

    def test_model(self):
        with torch.no_grad():
            # モデルを評価モードへ
            self.encoder.eval()
            self.decoder.eval()
            predicts = []
            row = []

            print('Evaluating......')
            with tqdm(total = len(self.test_data_loader), leave=False) as bar:
                for iters in self.test_data_loader:
                    bar.update(1)
                    time.sleep(0.001)
                    x, l = iters[0], iters[1]
                    x = x.to(self.device)
                    l = l.to(self.device)
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
                        inp_ = [self.text_vocab.get_itos()[idx] for idx in inp]
                        y = [self.label_vocab.get_itos()[idx] for idx in output]
                        p = [self.label_vocab.get_itos()[idx.item()] for idx in predict]
                        x_str = " ".join(inp_).replace('<pad> ', '')
                        y_str = " ".join(y[1:]).replace('<pad>', '')
                        p_str = " ".join(p).replace('<pad>', '')

                        judge = "O" if y_str == p_str else "X"
                        row.append([x_str, y_str, p_str, judge])

                
        predict_df = pd.DataFrame(row, columns=["input", "answer", "predict", "judge"])
        pd.set_option('display.max_columns', 4)
        pd.set_option('display.max_colwidth', 100)

        # 正解率を表示
        print("Accuracy : ", len(predict_df.query('judge == "O"')) / len(predict_df))

        print(predict_df.query('judge == "O"').head(10))
        # 間違えたデータを一部見てみる
        print(predict_df.query('judge == "X"').head(20))

    def plot_attention_map(self):
        # attention 可視化
        iter_ = iter(self.test_data_loader)
        for num in range(10000):
            iters = iter_.__next__()
            x, l = iters[0], iters[1]
            x = x.to(self.device)
            l = l.to(self.device)
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
                                    columns=[self.text_vocab.get_itos()[idx.item()] for idx in x[i][offset:]], 
                                    index=[self.label_vocab.get_itos()[idx.item()] for idx in torch.max(decoder_output[i],1)[1]])
                    plt.figure(figsize=(12, 12)) 
                    plt.ion()
                    sns.set_context("paper", 2.5)
                    sns.heatmap(df, xticklabels = 1, yticklabels = 1, square=True, linewidths=.3, cbar_kws = dict(use_gridspec=False,location="top"))
                    plt.yticks(rotation=0)
                    plt.draw()
                    plt.pause(0.0001)
            _ = input("Enter to the next, q -> quit")
            if _ == 'q':
                break
            plt.close(2)                    # 2番目のウインドウを閉じる
            plt.close(3)                    # 3番目のウインドウを閉じる



if __name__ == "__main__":
    command_analyzer = CommandAnalyzer()
    command_analyzer.train_model()
    if command_analyzer.is_test_model == True:
        command_analyzer.test_model()
        command_analyzer.plot_attention_map()
