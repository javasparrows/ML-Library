"""
分類の実行
"""
# では、学習したモデルを使って、ニュースが正しく分類できるかどうかを確認します。
# 今回はsports-watchディレクトリにあるsports-watch-4764756.txtを入力として分類を行います。

import os
import torch
from transformers import BertForSequenceClassification, BertJapaneseTokenizer

# 学習済みモデルの読み込み
model_dir = "./model"
loaded_model = BertForSequenceClassification.from_pretrained(model_dir)
# loaded_model.cuda() # cudaを使う場合は、この行を有効にする
loaded_tokenizer = BertJapaneseTokenizer.from_pretrained(model_dir)

# 分類するデータの読み込み
file = "./text/sports-watch/sports-watch-4764756.txt"  # sports-watchの適当なニュース

with open(file, "r") as f:
  sample_text = f.readlines()[3:]
  sample_text = "".join(sample_text)
  sample_text = sample_text.translate(str.maketrans({"\n":"", "\t":"", "\r":"", "\u3000":""})) 

max_length = 512
words = loaded_tokenizer.tokenize(sample_text)
word_ids = loaded_tokenizer.convert_tokens_to_ids(words)  # 単語をインデックスに変換
word_tensor = torch.tensor([word_ids[:max_length]])  # Tensorに変換

# 予測の実行
# word_tensor.cuda()  # cudaを使う場合は、この行を有効にする
y = loaded_model(word_tensor)  # 結果の予測
pred = y[0].argmax(-1)  # 最大値のインデックス（ディレクトリの番号）

# 結果の標準
path = "./text"
dir_files = os.listdir(path=path)
dirs = [f for f in dir_files if os.path.isdir(os.path.join(path, f))]  # ディレクトリ一覧
print("結果は", dirs[pred])