"""
学習
"""

# 先ほど作成したデータを入力とし、BERTを使ってニュースのカテゴリ分類を学習（fine-tuning）させるプログラムtrain.pyを作成します。

# モデル、トークナイザーの読み込み
# transformersに含まれている文章を分類するためのモデルBertForSequenceClassification、日本語を形態素解析するためのトークナイザーBertJapaneseTokenizerを読み込みます。

# cl-tohoku/bert-base-japanese-whole-word-maskingは事前学習済みの日本語BERTモデルです。
# このモデルは、東北大学の乾研究室によって作成されたもので、こちらのページで公開されています。
# https://www.nlp.ecei.tohoku.ac.jp/news-release/3284/

# BertForSequenceClassification、BertJapaneseTokenizerを読み込んだ際に、自動的にダウンロードされるため、あらかじめダウンロードをする必要はありません。


from transformers import BertForSequenceClassification, BertJapaneseTokenizer

# モデル
model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=9)
# model.cuda() # cudaを使う場合は、この行を有効にする
# トークナイザー
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")



"""
学習用、テスト用データの読み込み
"""
# 学習の入力データとして、先ほど保存したデータを読み込みます。

import os
from datasets import load_dataset

# トークナイズ用関数
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)
    
data_path = "./data"

# 学習用データ
train_data = load_dataset("csv", data_files=os.path.join(data_path, "news_train.csv"), column_names=["text", "label"], split="train")
train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
train_data.set_format("torch", columns=["input_ids", "label"])

# テスト用データ
test_data = load_dataset("csv", data_files=os.path.join(data_path, "news_test.csv"), column_names=["text", "label"], split="train")
test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))
test_data.set_format("torch", columns=["input_ids", "label"]) 



"""
Trainerの初期化
"""
# Trainerに、学習対象のモデル、学習用パラメーター、評価用関数、学習用データ、評価用データを設定して初期化します。

# 評価用関数
from sklearn.metrics import accuracy_score

def compute_metrics(result):
    labels = result.label_ids
    preds = result.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
    }


# Trainerの設定
from transformers import Trainer, TrainingArguments

# 学習用パラメーター
training_args = TrainingArguments(
    output_dir = "./results",
    num_train_epochs = 2,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 32,
    warmup_steps = 500,  # 学習係数が0からこのステップ数で上昇
    weight_decay = 0.01,  # 重みの減衰率
    # evaluate_during_training = True,  # ここの記述はバージョンによっては必要ありません
    logging_dir = "./logs",
)

# Trainerの初期化
trainer = Trainer(
    model = model, # 学習対象のモデル
    args = training_args, # 学習用パラメーター
    compute_metrics = compute_metrics, # 評価用関数
    train_dataset = train_data, # 学習用データ
    eval_dataset = test_data, # テスト用データ
)

# モデルの学習
# Trainerを使って、モデルの学習、評価を行います。
trainer.train() # 学習
trainer.evaluate() # 評価


# 学習済みモデルの保存
# 後で使うために、学習済みモデル、トークナイザーを保存します。
model_dir = "./model"
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)