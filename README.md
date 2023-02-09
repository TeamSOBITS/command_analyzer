# Command Analyzer Seq2Seq

GPSR(General Purpose Service Robot)タスクの命令理解リポジトリ

# 動作環境
pytorch <= 1.8
torchtext <= 0.9
※ 上記より新しいバージョンを利用する場合はブランチを切り替えて下さい

## インストール

```bash
$ cd ~/catkin_ws/src
$ git clone https://gitlab.com/TeamSOBITS/command_analyzer_seq2seq.git
$ cd seq2seq
$ bash install.sh
```

<注意>
  - install.sh でインストールするpytorchは使用するPCのGPU，CUDAによってバージョンが異なります
    - 手順通りにインストールして下さい


<br>

## データセットの作り方
データセットの作り方は[こちら]()から

<br>

## 学習の回し方
学習の回し方は[こちら]()から

<br>

## 使い方

```bash
$ cd ~/catkin_ws/src/command_analyzer_seq2seq/scripts
$ python3 example.py
please input command >>                     # 命令文を入力する
```
 ### 解説
 - example.pyの内部ではCommandAnalyzerクラスのインスタンスが生成される
    - CommandAnalyzerクラスのpredict関数に入力(命令文)を渡すと，認識結果が辞書型で返される
<br>
<br>
### 出力情報の説明
<table>
    <tr>
        <th>キー値</th>
        <th>説明</th>
        <th>値の例</th>
    </tr>
    <tr>
        <td>task</td>
        <td>タスクの内容</td>
        <td>bring, follow, find</td>
    </tr>
    <tr>
        <td>target</td>
        <td>タスクを行う対象(物体または人)</td>
        <td>apple, Michael, me</td>
    </tr>
    <tr>
        <td>prep_T1</td>
        <td>場所に対応する対象の位置関係</td>
        <td>in, on, under</td>
    </tr>
    <tr>
        <td>location_T1</td>
        <td>対象のある(いる)場所</td>
        <td>table, chair, shelf</td>
    </tr>
    <tr>
        <td>prep_T2</td>
        <td>場所に対応する対象の位置関係</td>
        <td>in, on, under</td>
    </tr>
    <tr>
        <td>location_T2</td>
        <td>対象のある(いる)場所</td>
        <td>table, chair, shelf</td>
    </tr>
    <tr>
        <td>room_T</td>
        <td>対象のある(いる)部屋</td>
        <td>living kitchen, bedroom</td>
    </tr>
    <tr>
        <td nowrap>destination</td>
        <td>タスクでの対象の行先(人または場所)</td>
        <td>me, Michael, place​</td>
    </tr>
    <tr>
        <td>prep_D1</td>
        <td>場所に対応する行先の位置関係</td>
        <td>in, on, under</td>
    </tr>
    <tr>
        <td>location_D1</td>
        <td>行先の場所</td>
        <td>table, chair, shelf</td>
    </tr>
    <tr>
        <td>prep_D2</td>
        <td>場所に対応する行先の位置関係</td>
        <td>in, on, under</td>
    </tr>
    <tr>
        <td>location_D2</td>
        <td>行先の場所</td>
        <td>table, chair, shelf</td>
    </tr>
    <tr>
        <td>room_D</td>
        <td>行先の部屋</td>
        <td>living kitchen, bedroom</td>
    </tr>
    <tr>
        <td>WYS</td>
        <td>人に質問(回答)するときの内容</td>
        <td>What day is it today?</td>
    </tr>
    <tr>
        <td>FIND</td>
        <td>findタスクで見つける内容</td>
        <td>name, count, gesture​</td>
    </tr>
    <tr>
        <td>obj_option</td>
        <td>物体の補足情報</td>
        <td>largest, heaviest, thinnest</td>
    </tr>
    <tr>
        <td>obj_num</td>
        <td>対象物体の数</td>
        <td>2, 3</td>
    </tr>
    <tr>
        <td>guesture</td>
        <td>人の補足情報</td>
        <td>waving left arm, sitting</td>
    </tr>
    <tr>
        <td>room_F</td>
        <td>最終目的地の部屋</td>
        <td>living kitchen, bedroom</td>
    </tr>
</table>
