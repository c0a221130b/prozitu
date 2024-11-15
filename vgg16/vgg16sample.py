from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import keras.utils as image
import numpy as np
import sys

# 認識対象の画像を読み込み
input_filename = sys.argv[1] #コマンドライン引数で画像のファイルパスを指定
input_image = image.load_img(input_filename, target_size=(224, 224))
input = np.expand_dims(image.img_to_array(input_image), axis=0)

# モデルの定義と認識処理を実行
model = VGG16(weights='imagenet')
results = model.predict(preprocess_input(input))

# Kerasが提供するdecode_predictionsを用いて結果を出力
decoded_results = decode_predictions(results, top=5)[0]
for decoded_result in decoded_results:
    print(decoded_result)