# coding:utf-8
import warnings
warnings.filterwarnings('ignore')           # 警告文の無視
import sys
sys.append("..")
import os
import re
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt # グラフ表示用のライブラリ
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torchtext.vocab import FastText, GloVe         # word2vecの上位互換
from network import Encoder, AttentionDecoder
from lib import lists, dicts


class CommandAnalyzer():
    def __init__(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # パラメータ設定
        self.sen_length = 30
        self.output_len = 20
        self.batch_size = 1220
        self.wordvec_size = 300
        self.hidden_size = 650
        self.dropout = 0.5
        self.learning_rate = 0.001
        self.momentum=0
        self.max_grad = 0.25
        self.eval_interval = 20
        self.predict_unk = True
        self.show_attention_map = True

        # モデルのパス
        self.model_path = "gpsr_2013"
        self.dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        self.encoder_path = "{}/model/{}/encoder.pth".format(self.dir_path, self.model_path)
        self.decoder_path = "{}/model/{}/decoder.pth".format(self.dir_path, self.model_path)
        self.text_vocab_path = "{}/model/{}/text_vocab.pth".format(self.dir_path, self.model_path)
        self.label_vocab_path = "{}/model/{}/label_vocab.pth".format(self.dir_path, self.model_path)

        #辞書ベクトルの読み込み
        self.text_vocab = torch.load(self.text_vocab_path)
        self.label_vocab = torch.load(self.label_vocab_path)
        self.vocab_size = len(self.text_vocab.itos)
        self.label_size = len(self.label_vocab.itos)
        self.vectors=GloVe(dim=300)


        # モデルの生成
        self.encoder = Encoder(self.vocab_size, self.wordvec_size, self.hidden_size, self.dropout, self.text_vocab, self.vectors, self.predict_unk)
        self.decoder = AttentionDecoder(self.vocab_size, self.wordvec_size, self.hidden_size, self.dropout, self.batch_size, self.label_vocab)
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.decoder.load_state_dict(torch.load(self.decoder_path))
        self.encoder.to(self.device)                                    # GPUを使う場合
        self.decoder.to(self.device)                                    # GPUを使う場合
        self.criterion = nn.CrossEntropyLoss()                 # 損失の計算
        self.softmax = nn.Softmax(dim=1)


    def tokenize(self, s: str) -> list:
        s = s.lower()
        for p in lists.remove_words:
            s = s.replace(p, '')
        for p in dicts.replace_phrases.keys():
            s = s.replace(p, dicts.replace_phrases[p])
        s = s.replace("'s", "")
        s = re.sub(r" +", r" ", s).strip()
        return s.split()

    def preprocessing(self, s: str) -> str:
        if s in dicts.replace_words.keys():
            return dicts.replace_words[s]
        else:
            return s

    def get_max_index(self, decoder_output):
        results = []
        for h in decoder_output:
            results.append(torch.argmax(h))
        return torch.tensor(results, device=self.device).view(self.batch_size, 1)

    def predict(self, cmd_sen):
        with torch.no_grad():
            # モデルを評価モードへ
            self.encoder.eval()
            self.decoder.eval()
            sentence = ['<pad>' for i in range(self.sen_length)]
            cmd_sen = self.tokenize(self.preprocessing(cmd_sen))
            sentence[self.sen_length-len(cmd_sen):] = cmd_sen
            # print(sentence)
            x = [self.text_vocab.stoi[w] for w in sentence]
            x = torch.tensor(x*self.batch_size).view(self.batch_size, -1).to(self.device)
            hs, encoder_state = self.encoder(x, sentence)

            # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、"_"のtensorをバッチサイズ分作成
            start_char_batch = [[self.label_vocab["_"]] for _ in range(self.batch_size)]
            decoder_input_tensor = torch.tensor(start_char_batch, device=self.device)

            # 変数名変換
            decoder_hidden = encoder_state

            # バッチ毎の結果を結合するための入れ物を定義
            batch_tmp = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
            #print(batch_tmp.size())
            # (100,1)
            attention_weights = torch.zeros(self.sen_length, 1, dtype=torch.long, device=self.device)

            for _ in range(self.output_len - 1):
                #tqdm.write('hs : {}'.format(hs.size()))
                decoder_output, decoder_hidden, attention_weight = self.decoder(decoder_input_tensor, hs, decoder_hidden)
                # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
                decoder_input_tensor = self.get_max_index(decoder_output.squeeze())
                #tqdm.write('dec_in_tsr : {}'.format(decoder_input_tensor.size()))
                # バッチ毎の結果を予測順に結合
                batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)
                if self.show_attention_map:
                    # print(attention_weight[0].shape)
                    # print(attention_weights.shape)
                    attention_weights = torch.cat([attention_weights, attention_weight[0]], dim=1)


            # 最初のbatch_tmpの0要素が先頭に残ってしまっているのでスライスして削除
            predict_list = [self.label_vocab.itos[idx.item()] for idx in batch_tmp[:,1:][0]]
            result = dicts.result_dict

            if self.show_attention_map:
                for offset, xt in enumerate(x[0]):
                    #print(xt)
                    if xt != 1:
                        break
                # print(offset)
                # print("size : {}".format(len(predict_list)))
                df = pd.DataFrame(data=torch.transpose(attention_weights[:,1:][offset:], 0, 1).cpu().numpy(), 
                                columns=[self.text_vocab.itos[idx.item()] for idx in x[0][offset:]], 
                                index=[predict_list])
                plt.figure(figsize=(12, 12)) 
                plt.ion()
                sns.set_context("paper", 2.5)
                sns.heatmap(df, xticklabels = 1, yticklabels = 1, square=True, linewidths=.3, cbar_kws = dict(use_gridspec=False,location="top"))
                plt.yticks(rotation=0)
                plt.draw()
                plt.pause(0.0001)
                _ = input("Enter to the close")
                plt.close()
            
            for res, pre in zip(result.keys(), predict_list):
                result[res] = pre
            return result


if __name__ == "__main__":
    command_analyzer = CommandAnalyzer()
    # print('Evaluating......')
    coke = ["cola", "coca", "coca-cola", "pepsi", "cocacola"]
    beer = ["budweiser", "guinness", "stout", "carlsberg", "heineken"]
    cereal = ["oatmeal", "granola", "bran", "oats", "cornflakes"]
    
    while True:
        try:
            # input_str = input("please input command >>").split()
            # input_str = input("please input command >>")
            input_str = "bring me the carlsberg in the living room"
            # input_str = "bring the coke on the cabinet to Michael"
            print(input_str)
            result =command_analyzer.predict(input_str)
            print(result)
            
            # for w in cereal:
            #     input_str = "bring me the {} in the living room".format(w)
            #     print(input_str)
            #     result =command_analyzer.predict(input_str)
            #     print(result)

            break
        except KeyboardInterrupt:
            break
