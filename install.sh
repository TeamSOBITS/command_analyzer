#!/bin/bash


read -p $'pytorch のCPU版をインストールします。
        \e[33m＊既にpytorchをインストールしている,
        　あるいはGPU版を入れたい人は実行しないで下さい。\e[0m
実行しますか? (y/N): ' yn
case "$yn" in
  [yY]*) sudo apt install python3-pip
         python3 -m pip install -U pip
         python3 -m pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 -f https://download.pytorch.org/whl/torch_stable.html;;
  *) echo "終了します。
pytorchのGPU版を入れたい方はこちら
https://pytorch.org/";;
esac

echo "---------------------------------------------------------------"
# torchtextのインストール

# torchのバージョン確認
version=`python3 -m pip freeze | grep torch= `
version=${version##*=}
version=${version%+*}
version=""

# 連想配列の定義
declare -A torch_text_versions=(
        ["1.13.1"]="0.14.1"
        ["1.13.0"]="0.14.0"
        ["1.12.0"]="0.13.0"
        ["1.11.0"]="0.12.0"
        ["1.10.0"]="0.11.0"
        ["1.9.1"]="0.10.1"
        ["1.9"]="0.10"
        ["1.8"]="0.9"
        ["1.7.1"]="0.8.1"
        ["1.7"]="0.8"
        ["1.6"]="0.7"
        ["1.5"]="0.6"
        ["1.4"]="0.5"
)

if [ -z "${torch_text_versions[$version]}" ]; then
  echo "pytorch == $version のバージョンに，このブランチは対応していません．
  ブランチを切り替えるか，pytorchのインストールをし直して下さい．
  pytorchのGPU版を入れたい方はこちら
  https://pytorch.org/"
  exit
fi
read -p $'ここから先の実行は\e[33mpytorchを入れた方のみ\e[0m行って下さい。
実行しますか? (y/N): ' yn
case "$yn" in
  [yY]*) python3 -m pip install torchtext==${torch_text_versions[$version]} numpy matplotlib tqdm pandas seaborn;;
  *) echo "終了します。
pytorchのGPU版を入れたい方はこちら
https://pytorch.org/";;
esac

# python3 -m pip install torchtext==${torch_text_versions[$version]}
# python3 -m pip install pandas seaborn

