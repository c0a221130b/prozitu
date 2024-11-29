import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np
import random

input_dim = 2                # 入力データの次元数：実数値1個なので1を指定
output_dim = 2               # 出力データの次元数：同上
num_hidden_units = 128       # 隠れ層のユニット数
len_sequence = 222            # 時系列の長さ
batch_size = 10              # ミニバッチサイズ
num_of_training_epochs = 100 # 学習エポック数
learning_rate = 0.001        # 学習率
num_training_samples = 1000  # 学習データのサンプル数

# 乱数シードを固定値で初期化
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

samples_x = np.loadtxt('mouse_x.csv', delimiter=',')
samples_label_x = samples_x[:, 0].astype(int)
samples_data_x = samples_x[:, 1:]
x_max_len = len(samples_data_x[0])

samples_y = np.loadtxt('mouse_y.csv', delimiter=',')
samples_label_y = samples_y[:, 0].astype(int)
samples_data_y = samples_y[:, 1:]
y_max_len = len(samples_data_y[0])


samples_data = np.stack([samples_data_x, samples_data_y], axis=2)
XY = samples_data.reshape((40, x_max_len, 2))

samples_label = np.stack([samples_label_x, samples_label_y], axis=1)
t = samples_label.reshape((40,1,2))



# モデル構築
model = Sequential()
model.add(LSTM(
    num_hidden_units,
    input_shape=(x_max_len, input_dim),
    return_sequences=False))
model.add(Dense(output_dim))
model.compile(loss="mean_squared_error", optimizer=Adam(lr=learning_rate))
model.summary()

# 学習
model.fit(
    XY, t,
    batch_size=batch_size,
    epochs=num_of_training_epochs,
    validation_split=0.1
)

# 予測
# (サンプル, 時刻, 特徴量の次元) の3次元の入力を与える。
tests_x = np.loadtxt('mouse_x_test.csv', delimiter=',')
tests_label_x = tests_x[:, 0].astype(int)
tests_data_x = tests_x[:, 1:]
tests_data_x_len = len(tests_data_x[0])

tests_y = np.loadtxt('mouse_y_test.csv', delimiter=',')
tests_label_y = tests_y[:, 0].astype(int)
tests_data_y = tests_y[:, 1:]
tests_data_y_len = len(tests_data_y[0])

tests_data = np.stack([tests_data_x, tests_data_y], axis=2)
test = tests_data.reshape((10, tests_data_x_len, 2))

print(model.predict(test)) # [[7.7854743]]
