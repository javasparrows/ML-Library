# 機械学習ライブラリ

## 手法名

### 🐤 多変量解析

| 手法名               | 場所 | 実装 |
| :------------------- | :--- | :--- |
| 線形回帰分析         | -    | -    |
| 主成分分析           | -    | -    |
| 因子分析             | -    | -    |
| 多次元尺度構成法     | -    | -    |
| 階層的クラスタリング | -    | -    |

### 📈 統計モデリング

| 手法名              | 場所 | 実装 |
| :------------------ | :--- | :--- |
| 一般化線形モデル    | -    | -    |
| 混合効果モデル      | -    | -    |
| 階層ベイズモデル    | -    | -    |
| 時系列解析 - AR     | -    | -    |
| 時系列解析 - MA     | -    | -    |
| 時系列解析 - ARIMA) | -    | -    |
| 状態空間モデル      | -    | -    |

### ⚙️ 機械学習

| 手法名                              | 場所 | 実装 |
| :---------------------------------- | :--- | :--- |
| サポートベクターマシン              | -    | -    |
| ランダムフォレスト                  | -    | -    |
| 勾配ブースティング決定木 - LightGBM | -    | -    |
| 勾配ブースティング決定木 - XGBoost  | -    | -    |
| 勾配ブースティング決定木 - CatBoost | -    | -    |
| k 近傍法                            | -    | -    |
| k-means                             | -    | -    |
| LDA (Latent Dirichlet Allocation)   | -    | -    |
| 多層パーセプトロン                  | -    | -    |

### 🩻 深層学習 - CNN

| 手法名                                                                                       | 年        | 著者               | 場所 | 実装 |
| :------------------------------------------------------------------------------------------- | :-------- | :----------------- | :--- | :--- |
| [Neocognition](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf)                   | 1980      | Fukushima & Miyake | -    |
| [LeNet-5](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)                      | 1989-1998 | LeCun et al.       | -    |
| [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 2012      | Krizhevsky et al.  | -    |
| [ZFNet](https://arxiv.org/abs/1311.2901)                                                     | 2013      | Zeiler & Fergus    | -    |
| [VGGNet](https://arxiv.org/abs/1409.1556)                                                    | 2014      | Simonyan et al.    | -    |
| [GoogLeNet](https://arxiv.org/abs/1409.4842)                                                 | 2014      | C. Szegedy et al.  | -    |
| [ResNet](https://arxiv.org/abs/1512.03385)                                                   | 2015      | He et al.          | -    |
| ZFNet                                                                                        | 2013      | -                  | -    |
| ZFNet                                                                                        | 2013      | -                  | -    |

### 🕒 深層学習 - RNN

### 深層学習 - CNN

### 👀 深層学習 - Transformer

| 手法名                                                           | 場所 | 実装 |
| :--------------------------------------------------------------- | :--- | :--- |
| [Vision Transformer](https://openreview.net/forum?id=YicbFdNTTy) | -    | -    |
|                                                                  | -    | -    |
|                                                                  | -    | -    |

### データセット

### 画像

- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

### 医療系

- [ChestX-ray8](https://paperswithcode.com/dataset/chestx-ray8)
- [TensorFlow patch_camelyon Medical Images](https://www.tensorflow.org/datasets/catalog/patch_camelyon)
- [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
- [Recursion Cellular Image Classification](https://www.kaggle.com/datasets/xhlulu/recursion-cellular-image-classification-224-jpg)

# 環境構築

## fish & fisher

`$ brew install fish`

`$ curl -sL https://git.io/fisher | source && fisher install jorgebucaran/fisher`

Plugin をインストール

`$ fisher install simnalamburt/shellder`

`$ fisher install edc/bass`

`$ fisher install jethrokuan/fzf`

`$ fisher install jorgebucaran/nvm.fish`

`$ fisher list`

を実行すると以下になる

```
jorgebucaran/fisher
simnalamburt/shellder
edc/bass
jethrokuan/fzf
jorgebucaran/nvm.fish
```

## [Poetry](https://python-poetry.org/docs/)

`$ curl -sSL https://install.python-poetry.org | python3 -`

`$ poetry config virtualenvs.in-project true`

`$ poetry init`

`$ poetry install`

`$ bash`

もし `requirements.txt` があったら
`$ for package in $(cat requirements.txt); do poetry add "${package}"; done`

`$ poetry shell` or `$ poetry run python ***.py`で仮想環境内で Python を実行できる

# データセット準備

## CIFAR-10

`$ cd dataset`

`$ bash download_cifar10.sh`
