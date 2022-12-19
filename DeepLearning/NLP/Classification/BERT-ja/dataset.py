# https://www.rondhuit.com/download.html#ldcc
# wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
# tar -zxvf ldcc-20140209.tar.gz


# 下記のプログラムでは、ニュースの各ファイルの本文を抽出し、改行、全角スペース、タブを除去して、1行の文に変換したのちに、カテゴリの番号（0-9）とのセットにします。


import glob
import os

raw_data_path = "./text"  # ライブドアニュースを格納したディレクトリ

dir_files = os.listdir(path=raw_data_path)
dirs = [f for f in dir_files if os.path.isdir(os.path.join(raw_data_path, f))]
text_label_data = []  # 文章とラベル（カテゴリ）のセット

for i in range(len(dirs)):
    dir = dirs[i]
    files = glob.glob(os.path.join(raw_data_path, dir, "*.txt"))

    for file in files:
        if os.path.basename(file) == "LICENSE.txt": # 各ディレクトリにあるLICENSE.txtを除外する
            continue

        with open(file, "r") as f:
            text = f.readlines()[3:]
            text = "".join(text)
            text = text.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""})) 
            text_label_data.append([text, i])


# 学習用、テスト用データの作成、保存
# 先ほど作成した、本文、ラベル（カテゴリ）のセットを、学習用、評価用のデータに分割し、CSVファイル（news_train.csv、news_test.csv）として保存します。

import csv
from sklearn.model_selection import train_test_split

news_train, news_test =  train_test_split(text_label_data, shuffle=True)  # データを学習用とテスト用に分割
data_path = "./data"

with open(os.path.join(data_path, "news_train.csv"), "w") as f:
  writer = csv.writer(f)
  writer.writerows(news_train) # 

with open(os.path.join(data_path, "news_test.csv"), "w") as f:
  writer = csv.writer(f)
  writer.writerows(news_test)