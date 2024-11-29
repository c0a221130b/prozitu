import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np
import random

input_dim = 1                # 入力データの次元数：実数値1個なので1を指定
output_dim = 1               # 出力データの次元数：同上
num_hidden_units = 128       # 隠れ層のユニット数
len_sequence = 64            # 時系列の長さ
batch_size = 10              # ミニバッチサイズ
num_of_training_epochs = 100 # 学習エポック数
learning_rate = 0.001        # 学習率
num_training_samples = 1000  # 学習データのサンプル数

# 乱数シードを固定値で初期化
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

samples = np.loadtxt('mouse_x.csv', delimiter=',')
samples_label = samples[:, 0].astype(int)
samples_data = samples[:, 1:65]
t = samples_label
X = samples_data.reshape((20, 64, 1))

# モデル構築
model = Sequential()
model.add(LSTM(
    num_hidden_units,
    input_shape=(len_sequence, input_dim),
    return_sequences=False))
model.add(Dense(output_dim))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
model.summary()

# 学習
model.fit(
    X, t,
    batch_size=batch_size,
    epochs=num_of_training_epochs,
    validation_split=0.1
)

# 予測
# (サンプル, 時刻, 特徴量の次元) の3次元の入力を与える。
tests = np.loadtxt('mouse_x_test.csv', delimiter=',')
tests_label = tests[:, 0].astype(int)
tests_data = tests[:, 1:65]
test = tests_data.reshape((20, 64, 1))

print(model.predict(test)) # [[7.7854743]]
