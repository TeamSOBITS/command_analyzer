#coding:utf-8
import sys
sys.path.append('..')  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import torch
import torch.nn as nn


# deviceの設定(GPUを使う場合)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100, dropout_ratio=0.5, vocab=None, vocab_vectors=None, vectors=None, is_predict_unk=True):
        super(Encoder, self).__init__()
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.vocab = vocab
        self.vectors = vectors
        self.vocab_vectors = vocab_vectors
        self.is_predict_unk = is_predict_unk

        # レイヤの生成
        # self.embed = nn.Embedding(V, D, padding_idx=vocab.get_stoi()['<pad>'])
        self.embed = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, H, num_layers=1, bias=True, batch_first=True, dropout=dropout_ratio)
        self.affine = nn.Linear(H, V, bias=True)
        self.dropout = nn.Dropout(dropout_ratio)

        # 重みの初期化
        self.embed.weight = nn.Parameter(self.vocab_vectors)
        self.embed.weight.requires_grad = False
        # nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.normal_(self.lstm.weight_ih_l0, std=1/np.sqrt(D))
        nn.init.normal_(self.lstm.weight_hh_l0, std=1/np.sqrt(H))
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        # self.affine.weight = self.embed.weight      # 重みの共有
        # nn.init.zeros_(self.affine.bias)


    def cosine_matrix(self, a, b):
        dot = torch.matmul(a, torch.t(b))
        norm = torch.matmul(torch.norm(a, dim=0).unsqueeze(-1), torch.norm(b, dim=0).unsqueeze(0))
        return dot / norm

    def compare_word_sim(self, a, b):
        if torch.norm(b) == 0:
            return torch.norm(b)
        norm = torch.norm(a-b)
        cos_sim = torch.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        # return cos_sim / norm
        return cos_sim

    def forward(self, xs, sentence=None):
        xvecs = self.embed(xs)
        if self.is_predict_unk == True:
            for i, x in enumerate(xs[0]):
                if x.item() == self.vocab.get_stoi()['<unk>']:
                    print("unknown sentence :",sentence[i])
                    max_var = -1
                    synonym = ""
                    for w in self.vocab.get_itos():
                        if w in ["<pad>", "<unk>"]:
                            continue
                        vec_w = self.vectors.get_vecs_by_tokens(w, lower_case_backup=True)
                        vec_sen = self.vectors.get_vecs_by_tokens(sentence[i], lower_case_backup=True)
                        # print("w :", vec_w.shape)
                        # print("sen :", vec_sen.shape)
                        var = self.cosine_matrix(vec_w, vec_sen)
                        # var = self.compare_word_sim(vec_w, vec_sen)
                        if w == "":
                            print("w : {}\nmin_var : {}".format(w, var))
                        if var > max_var:
                            print("w : {}\nmin_var : {}".format(w, var))
                            max_var = var 
                            synonym = w
                    # print("w : {}\nmin_var : {}".format(synonym, max_var))
                    xvecs[0][i] = self.embed(torch.tensor([self.vocab.get_stoi()[synonym]]).to("cuda:0"))
                    # print(xvecs)
        xs, h = self.lstm(xvecs)
        # score = self.affine(xs)

        return xs, h


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100, dropout_ratio=0.5, batch_size=100, vocab=None):
        super(AttentionDecoder, self).__init__()
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.H = H
        self.batch_size = batch_size

        # レイヤの生成
        self.embed = nn.Embedding(V, D)
        self.lstm = nn.LSTM(D, H, num_layers=1, bias=True, batch_first=True, dropout=dropout_ratio)
        self.affine = nn.Linear(H * 2, V, bias=True)        # H * 2 : 各系列のLSTMの隠れ層とAttention層で計算したコンテキストベクトルをtorch.catでつなぎ合わせることで長さが２倍になるため
        self.dropout = nn.Dropout(dropout_ratio)
        self.softmax = nn.Softmax(dim=1)                    # 列方向

        # 重みの初期化
        # self.embed.weight = nn.Parameter(vocab.vectors)
        # self.embed.weight.requires_grad = False
        nn.init.normal_(self.embed.weight, std=0.01)
        nn.init.normal_(self.lstm.weight_ih_l0, std=1/np.sqrt(D))
        nn.init.normal_(self.lstm.weight_hh_l0, std=1/np.sqrt(H))
        nn.init.zeros_(self.lstm.bias_ih_l0)
        nn.init.zeros_(self.lstm.bias_hh_l0)
        nn.init.normal_(self.affine.weight, std=1/np.sqrt(H))
        nn.init.zeros_(self.affine.bias)


    def forward(self, xs, hs, h):
        xs = self.embed(xs)
        output, state = self.lstm(xs, h)
        # print("hs size : ", hs.size())      # hs.size() = ([batch_size, input_size, hidden_size])
        # print("output size : ", output.size())  # output.size() = ([batch_size, output_size, hidden_size])

        # Attention層        
        
        # 列の変換
        t_output = torch.transpose(output, 1, 2)    # t_output.size() = ([batch_size, hidden_size, output_size])
        # print("t_output size : ", t_output.size())

        # bmm(batch marix * matrix)でバッチも考慮してまとめて行列計算
        s = torch.bmm(hs, t_output) # s.size() = ([batch_size, input_size, output_size])
        # print("s size : ", s.size())

        # 列方向(dim=1)でsoftmaxをとって確率表現に変換
        # この値を後のAttentionの可視化などにも使うため、returnで返しておく
        attention_weight = self.softmax(s) # attention_weight.size() = ([100, 29, 10])
        # print('a_weight : ', attention_weight.size())

        # コンテキストベクトルをまとめるために入れ物を用意
        c = torch.zeros(self.batch_size, 1, self.H, device=device) # c.size() = ([batch_size, 1, hidden_size])

        # 各DecoderのLSTM層に対するコンテキストベクトルをまとめて計算する方法がわからなかったので、
        # 各層（Decoder側のLSTM層は生成文字列が19文字なので19個ある）におけるattention weightを取り出してforループ内でコンテキストベクトルを１つずつ作成する
        # バッチ方向はまとめて計算できたのでバッチはそのまま
        for i in range(attention_weight.size()[2]): # 19回ループ

            # attention_weight[:,:,i].size() = ([100, 29])
            # i番目のGRU層に対するattention weightを取り出すが、テンソルのサイズをhsと揃えるためにunsqueezeする
            unsq_weight = attention_weight[:,:,i].unsqueeze(2) # unsq_weight.size() = ([batch_size, output_size, 1])
            # print("u_weghit : ", unsq_weight.size())

            # hsの各ベクトルをattention weightで重み付けする
            weighted_hs = hs * unsq_weight # weighted_hs.size() = ([batch_size, input_size, hidden_size])
            # print("w_hs", weighted_hs.size())

            # attention weightで重み付けされた各hsのベクトルをすべて足し合わせてコンテキストベクトルを作成
            weight_sum = torch.sum(weighted_hs, axis=1).unsqueeze(1) # weight_sum.size() = ([batch_size, 1, hidden_size])
            # print("w_sum : ", weight_sum.size())

            # print('c : ', c.size())
            c = torch.cat([c, weight_sum], dim=1) # c.size() = ([batch_size, i, hidden_size])

        # 箱として用意したzero要素が残っているのでスライスして削除
        c = c[:,1:,:]

        # hidden_size*2 : 各系列のLSTMの隠れ層とAttention層で計算したコンテキストベクトルをつなぎ合わせることで長さが２倍になるため
        output = torch.cat([output, c], dim=2) # output.size() = ([batch_size, output_size, hidden_size*2])
        # print("output_1", output.size())
        output = self.affine(output) # output.size() = ([batch_size, output_size, label_size])
        # print("output_2", output.size())

        return output, state, attention_weight





