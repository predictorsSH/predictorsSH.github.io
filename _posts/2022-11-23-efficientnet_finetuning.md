---
layout: post
title: Image classification via fine-tuning with EfficientNet
subtitle: keras example - EfficientNet Fine-tuning
author: san9hyun
categories: keras-example
banner : /assets/images/banners/solar-system.jpg
tags: EfficientNet Fine-tuning transfer-learning
---
**dark 모드로 보기** <br>

[keras code example](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) 을 따라 공부한 것 입니다.


## Description

EfficientNet을 사용한 개 사진 분류<br>
사전학습된 가중치를 활용하지 않을때와, 활용하였을때를 비교


## Introduction:what is EfficientNet

EfficinetNet은 2019년 [논문](https://arxiv.org/abs/1905.11946)에서 소개되었다.
<br>이미지 분류와 전이학습 모두에서 SOTA에 도달한 모델들중 하나이다.


## 케라스 EfficientNet 구현

TF2.3 이후로 EfficientNet이 tf.keras와 함께 제공된다.<br>

```python
from tensorflow.keras.applications import EfficientNetB0
moidel = EfficientNetB0(weights='imagenet')
```

위 모델은 (224,224,3)크기의 이미지를 입력으로 받고, 입력 값의 크기는 [0,255] 범위이다.<br> Normalization은 모델 내부에 포함되어있다.

ImageNet으로 EfficientNet을 학습시키는 것은 매우 큰 리소스와,<br>
모델 아키텍처에는 포함되어있지 않은 몇몇 기술들을 필요로 한다.
따라서 Keras의 구현은 기본적으로 사전학습된 가중치를 로드한다.




아래 B0에서 B7까지 모델들의 입력 shape은 전부 다르다.<br>


| Base model | resolution|
|----------------|-----|
| EfficientNetB0 | 224 |
| EfficientNetB1 | 240 |
| EfficientNetB2 | 260 |
| EfficientNetB3 | 300 |
| EfficientNetB4 | 380 |
| EfficientNetB5 | 456 |
| EfficientNetB6 | 528 |
| EfficientNetB7 | 600 |



전이학습을 하고자 할때, Keras는 최상위 레이어를 제거하는 옵션을 제공한다.

```python
model = EfficientNetB0(include_top=False, weights='imagenet')
```

이 옵션은 마지막 Dense 레이어를 제외한다.<br>
마지막 레이어를 사용자 정의 레이어로 대체하면, EfficientNet을 특징 추출기로 활용할 수 있게된다.

drop_connect_rate는 학습시 확률적으로 layer를 스킵하는 옵션이다.<br>
해당 옵션은 finetuning에서 정규화를 돕는다. 로드된 가중치에는 영향을 미치지 않는다.
```python
model = EfficientNetB0(weights='imagenet', drop_connect_rate=0.4)
```

## 예제 : Staford Dog분류를 위한 EfficientNetB0


```python
IMG_SIZE = 224
```

## Setup and Data loading


```python

import tensorflow_datasets as tfds

batch_size = 64

dataset_name = "stanford_dogs"
(ds_train, ds_test), ds_info = tfds.load(
    dataset_name, split=["train","test"], with_info=True, as_supervised=True
)

NUM_CLASSES = ds_info.features["label"].num_classes
```

    Downloading and preparing dataset 778.12 MiB (download: 778.12 MiB, generated: Unknown size, total: 778.12 MiB) to ~/tensorflow_datasets/stanford_dogs/0.2.0...


    Dl Completed...: 0 url [00:00, ? url/s]


    Dl Size...: 0 MiB [00:00, ? MiB/s]


    Dl Completed...: 0 url [00:00, ? url/s]


    Dl Size...: 0 MiB [00:00, ? MiB/s]


    Extraction completed...: 0 file [00:00, ? file/s]


    Generating splits...:   0%|          | 0/2 [00:00<?, ? splits/s]


    Generating train examples...:   0%|          | 0/12000 [00:00<?, ? examples/s]


    Shuffling ~/tensorflow_datasets/stanford_dogs/0.2.0.incompleteG6P1KZ/stanford_dogs-train.tfrecord*...:   0%|  …


    Generating test examples...:   0%|          | 0/8580 [00:00<?, ? examples/s]


    Shuffling ~/tensorflow_datasets/stanford_dogs/0.2.0.incompleteG6P1KZ/stanford_dogs-test.tfrecord*...:   0%|   …


    Dataset stanford_dogs downloaded and prepared to ~/tensorflow_datasets/stanford_dogs/0.2.0. Subsequent calls will reuse this data.


이미지가 다양한 크기를 가지고 있을때. 우리는 크기를 조정해야한다.


```python
import tensorflow as tf

size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image,label : (tf.image.resize(image,size), label))
ds_test = ds_test.map(lambda image,label : (tf.image.resize(image,size), label))
```

## 데이터 시각화


```python
import matplotlib.pyplot as plt

# 라벨에 대한 정보는 tfds를 로드할때 함께 가져온 ds_info.features['label']
def format_label(label):
  string_label = ds_info.features['label'].int2str(label)
  return string_label.split("-")[1]

for i, (image,label) in enumerate(ds_train.take(9)):
  ax = plt.subplot(3,3,i+1)
  plt.imshow(image.numpy().astype("uint8"))
  plt.title("{}".format(format_label(label)))
  plt.axis("off")
```


![png](/assets/images/contents/keras-examples/Image classification via fine-tuning with EfficientNet/Image classification via fine-tuning with EfficientNet_16_0.png)


## 데이터 증강

전처리 layers API를 활용하여 이미지 증강을 할 수 있다.<br>
아래 Sequential model은 분류 모델을 빌드할때 하나의 부품으로 사용할 수 있고<br>
분류 모델에 입력하기전, 데이터를 전처리하는 기능으로 활용할수도 있다.


```python
from tensorflow import keras
from keras import layers
from keras.models import Sequential


img_augmentation = Sequential(
    [
     layers.RandomRotation(factor=0.15),
    #  layers.RandomTranslation(height_factor=0.3, width_factor=0.3), # warning 발생
     layers.RandomFlip(),
    # layers.RandomContrast(factor=0.1), # warning 발생
    ],
    name = "img_augmentation"
)
```


```python
for image, label in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(tf.expand_dims(image, axis=0))
        plt.imshow(aug_img[0].numpy().astype("uint8"))
        plt.title("{}".format(format_label(label)))
        plt.axis("off")
```


![png](/assets/images/contents/keras-examples/Image classification via fine-tuning with EfficientNet/Image classification via fine-tuning with EfficientNet_20_0.png)


## 입력 데이터 준비

입력 데이터와 데이터 증강이 올바르게 동작하는지 확인하기 위해, 훈련 데이터 세트를 준비하자.<br>

입력 데이터의 크기는 IMG_SIZE로 표준화 되고 label은 one-hot 인코딩된다.


```python
def input_preprocess(image, label):
  label = tf.one_hot(label, NUM_CLASSES)
  return image, label

# num_parallel_calls : 병렬처리
ds_train = ds_train.map(input_preprocess, num_parallel_calls=tf.data.AUTOTUNE) 

ds_train = ds_train.batch(batch_size=batch_size, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

# 왜 test에서는 병렬처리와, prefetch를 사용하지 않는가?
ds_test = ds_test.map(input_preprocess)
ds_test = ds_test.batch(batch_size=batch_size, drop_remainder=True)
```

## 모델 학습

### 사전학습된 가중치를 활용하지 않음

원하는 데이터 세트에서 EfficientNet을 쉽게 교육할 수 있다.<br>
그러나 소규모 데이터 세트, 특히 해상도가 낮은 데이터 세트에서 EfficientNet을 교육하게되면 과대적합이라는 중대한 과제에 직면할 것이다.


```python
from tensorflow.python.distribute.distribute_lib import Strategy
from keras.applications import EfficientNetB0
import tensorflow as tf

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print("Device:", tpu.master())
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:
    print("Not connected to a TPU runtime. Using CPU/GPU strategy")
    strategy = tf.distribute.MirroredStrategy()

```

    Not connected to a TPU runtime. Using CPU/GPU strategy



```python
with strategy.scope():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    outputs = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES)(x)  # weights=None으로 사전학습된 가중치를 활용하지 않음

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

model.summary()

epochs = 40  
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     img_augmentation (Sequentia  (None, 224, 224, 3)      0         
     l)                                                              
                                                                     
     efficientnetb0 (Functional)  (None, 120)              4203291   
                                                                     
    =================================================================
    Total params: 4,203,291
    Trainable params: 4,161,268
    Non-trainable params: 42,023
    _________________________________________________________________
    Epoch 1/40
    187/187 [==============================] - 160s 753ms/step - loss: 4.9565 - accuracy: 0.0144 - val_loss: 5.0062 - val_accuracy: 0.0058
    Epoch 2/40
    187/187 [==============================] - 139s 744ms/step - loss: 4.5999 - accuracy: 0.0244 - val_loss: 4.9292 - val_accuracy: 0.0124
    Epoch 3/40
    187/187 [==============================] - 137s 735ms/step - loss: 4.4523 - accuracy: 0.0290 - val_loss: 4.8505 - val_accuracy: 0.0248
    Epoch 4/40
    187/187 [==============================] - 137s 731ms/step - loss: 4.3081 - accuracy: 0.0422 - val_loss: 4.6877 - val_accuracy: 0.0384
    Epoch 5/40
    187/187 [==============================] - 137s 732ms/step - loss: 4.1785 - accuracy: 0.0567 - val_loss: 4.6191 - val_accuracy: 0.0504
    Epoch 6/40
    187/187 [==============================] - 137s 735ms/step - loss: 4.0775 - accuracy: 0.0671 - val_loss: 4.7597 - val_accuracy: 0.0532
    Epoch 7/40
    187/187 [==============================] - 138s 737ms/step - loss: 3.9805 - accuracy: 0.0799 - val_loss: 6.0232 - val_accuracy: 0.0557
    Epoch 8/40
    187/187 [==============================] - 139s 741ms/step - loss: 3.8893 - accuracy: 0.0890 - val_loss: 4.4315 - val_accuracy: 0.0740
    Epoch 9/40
    187/187 [==============================] - 139s 742ms/step - loss: 3.8050 - accuracy: 0.1037 - val_loss: 4.6189 - val_accuracy: 0.0613
    Epoch 10/40
    187/187 [==============================] - 138s 740ms/step - loss: 3.7085 - accuracy: 0.1167 - val_loss: 4.0406 - val_accuracy: 0.0887
    Epoch 11/40
    187/187 [==============================] - 138s 740ms/step - loss: 3.6150 - accuracy: 0.1298 - val_loss: 4.6661 - val_accuracy: 0.0577
    Epoch 12/40
    187/187 [==============================] - 138s 740ms/step - loss: 3.5536 - accuracy: 0.1411 - val_loss: 4.0313 - val_accuracy: 0.0898
    Epoch 13/40
    187/187 [==============================] - 139s 741ms/step - loss: 3.4707 - accuracy: 0.1545 - val_loss: 3.6367 - val_accuracy: 0.1307
    Epoch 14/40
    187/187 [==============================] - 138s 740ms/step - loss: 3.3695 - accuracy: 0.1735 - val_loss: 3.9461 - val_accuracy: 0.1187
    Epoch 15/40
    187/187 [==============================] - 138s 738ms/step - loss: 3.2879 - accuracy: 0.1842 - val_loss: 3.6254 - val_accuracy: 0.1372
    Epoch 16/40
    187/187 [==============================] - 138s 738ms/step - loss: 3.1810 - accuracy: 0.2030 - val_loss: 4.1067 - val_accuracy: 0.1144
    Epoch 17/40
    187/187 [==============================] - 138s 737ms/step - loss: 3.0913 - accuracy: 0.2193 - val_loss: 3.3958 - val_accuracy: 0.1740
    Epoch 18/40
    187/187 [==============================] - 138s 736ms/step - loss: 3.0052 - accuracy: 0.2346 - val_loss: 3.4114 - val_accuracy: 0.1771
    Epoch 19/40
    187/187 [==============================] - 138s 739ms/step - loss: 2.9131 - accuracy: 0.2544 - val_loss: 3.4924 - val_accuracy: 0.1793
    Epoch 20/40
    187/187 [==============================] - 145s 777ms/step - loss: 2.8408 - accuracy: 0.2673 - val_loss: 3.5303 - val_accuracy: 0.1908
    Epoch 21/40
    187/187 [==============================] - 152s 811ms/step - loss: 2.8333 - accuracy: 0.2711 - val_loss: 3.6293 - val_accuracy: 0.1744
    Epoch 22/40
    187/187 [==============================] - 139s 741ms/step - loss: 2.6987 - accuracy: 0.2982 - val_loss: 3.6174 - val_accuracy: 0.1559
    Epoch 23/40
    187/187 [==============================] - 138s 736ms/step - loss: 2.6453 - accuracy: 0.3068 - val_loss: 3.1899 - val_accuracy: 0.2283
    Epoch 24/40
    187/187 [==============================] - 138s 739ms/step - loss: 2.5140 - accuracy: 0.3340 - val_loss: 3.2601 - val_accuracy: 0.2253
    Epoch 25/40
    187/187 [==============================] - 139s 741ms/step - loss: 2.4188 - accuracy: 0.3481 - val_loss: 3.2684 - val_accuracy: 0.2245
    Epoch 26/40
    187/187 [==============================] - 143s 763ms/step - loss: 2.3152 - accuracy: 0.3705 - val_loss: 3.5892 - val_accuracy: 0.2081
    Epoch 27/40
    187/187 [==============================] - 144s 767ms/step - loss: 2.2335 - accuracy: 0.3922 - val_loss: 3.5665 - val_accuracy: 0.2116
    Epoch 28/40
    187/187 [==============================] - 143s 762ms/step - loss: 2.1507 - accuracy: 0.4083 - val_loss: 3.4929 - val_accuracy: 0.2211
    Epoch 29/40
    187/187 [==============================] - 146s 779ms/step - loss: 2.0838 - accuracy: 0.4290 - val_loss: 3.5268 - val_accuracy: 0.2077
    Epoch 30/40
    187/187 [==============================] - 150s 804ms/step - loss: 1.9571 - accuracy: 0.4567 - val_loss: 3.5302 - val_accuracy: 0.2305
    Epoch 31/40
    187/187 [==============================] - 142s 757ms/step - loss: 1.8434 - accuracy: 0.4834 - val_loss: 3.8229 - val_accuracy: 0.2126
    Epoch 32/40
    187/187 [==============================] - 141s 753ms/step - loss: 1.7469 - accuracy: 0.5015 - val_loss: 3.9446 - val_accuracy: 0.1999
    Epoch 33/40
    187/187 [==============================] - 140s 750ms/step - loss: 1.6506 - accuracy: 0.5284 - val_loss: 3.9985 - val_accuracy: 0.2070
    Epoch 34/40
    187/187 [==============================] - 142s 758ms/step - loss: 1.5429 - accuracy: 0.5558 - val_loss: 4.8466 - val_accuracy: 0.1489
    Epoch 35/40
    187/187 [==============================] - 142s 759ms/step - loss: 1.4759 - accuracy: 0.5702 - val_loss: 4.2296 - val_accuracy: 0.2046
    Epoch 36/40
    187/187 [==============================] - 141s 754ms/step - loss: 1.3492 - accuracy: 0.6023 - val_loss: 4.2922 - val_accuracy: 0.2023
    Epoch 37/40
    187/187 [==============================] - 138s 740ms/step - loss: 1.3395 - accuracy: 0.6047 - val_loss: 4.3639 - val_accuracy: 0.2059
    Epoch 38/40
    187/187 [==============================] - 138s 737ms/step - loss: 1.1424 - accuracy: 0.6636 - val_loss: 4.2533 - val_accuracy: 0.2256
    Epoch 39/40
    187/187 [==============================] - 138s 738ms/step - loss: 1.0554 - accuracy: 0.6862 - val_loss: 4.5947 - val_accuracy: 0.2123
    Epoch 40/40
    187/187 [==============================] - 138s 736ms/step - loss: 0.9880 - accuracy: 0.7020 - val_loss: 4.7215 - val_accuracy: 0.2043



```python
import matplotlib.pyplot as pyplot

def plot_hist(hist):
  plt.plot(hist.history["accuracy"])
  plt.plot(hist.history["val_accuracy"])
  plt.title("model accuracy")
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  plt.legend(["train","validation"], loc="upper left")
  plt.show()

plot_hist(hist)
```


![png](/assets/images/contents/keras-examples/Image classification via fine-tuning with EfficientNet/Image classification via fine-tuning with EfficientNet_28_0.png)


모델 정확도가 매우 느리게 상승하는 것을 볼 수 있고, 검증 정확도는 해당 모델이 과대적합 되고 있음을 보여준다.

### 사전학습 가중치를 활용한 전이학습

최상위 레이어를 제외한 다른 모든 레이어를 동결하여(업데이트X) 전이학습 진행한다.<br>
이런 경우 상대적으로 큰 학습률(1e-2)을 사용할수 있다.<br>


```python
def build_model(num_classes):
  inputs = layers.Input(shape=(IMG_SIZE,IMG_SIZE,3))
  x = img_augmentation(inputs)
  model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")  # include_top=False 옵션으로 최상위 레이어(classifier) 제거 

  model.trainable = False                                                        # Freeze the pretrained weights, 사전학습된 가중치는 업데이트 하지 않음

  x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
  x = layers.BatchNormalization()(x)

  x = layers.Dropout(0.2, name="top_dropout")(x)
  outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

  model = keras.Model(inputs, outputs, name="EfficientNet")
  optimizer = keras.optimizers.Adam(learning_rate=1e-2)
  model.compile(
      optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
  )

  return model
```

이 모델에서 include_top=False 옵션을 사용했다.<br>
위에서 해당 옵션이 마지막 Dense 레이어를 제거한다고 설명하였는데, EfficientNet뒤에 Dense레이어만 추가로 이어 붙이는 것이 아니라
GlobalAveragePooling층을 생성하여 붙이는 것을 볼 수 있다.<br>
이는  include_top 옵션이 사실 마지막 Dense레이어만 아니라, GAP레이어, Dropout레이어, Dense레이어 이 세가지 레이어를 제거하는 것임을 알 수 있다.<br>
게시글 제일 아래 참고할 이미지가 있다.

```python
with strategy.scope():
  model = build_model(num_classes=NUM_CLASSES)

epochs = 25
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=1)
plot_hist(hist)
```

    Downloading data from https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5
    16705208/16705208 [==============================] - 0s 0us/step
    Epoch 1/25
    187/187 [==============================] - 59s 274ms/step - loss: 3.0716 - accuracy: 0.4476 - val_loss: 0.8643 - val_accuracy: 0.7435
    Epoch 2/25
    187/187 [==============================] - 49s 263ms/step - loss: 1.4652 - accuracy: 0.6220 - val_loss: 0.6771 - val_accuracy: 0.7963
    Epoch 3/25
    187/187 [==============================] - 49s 262ms/step - loss: 1.1839 - accuracy: 0.6654 - val_loss: 0.6558 - val_accuracy: 0.8003
    Epoch 4/25
    187/187 [==============================] - 49s 264ms/step - loss: 1.0481 - accuracy: 0.6945 - val_loss: 0.7162 - val_accuracy: 0.7898
    Epoch 5/25
    187/187 [==============================] - 50s 265ms/step - loss: 1.0021 - accuracy: 0.7073 - val_loss: 0.7262 - val_accuracy: 0.7830
    Epoch 6/25
    187/187 [==============================] - 51s 271ms/step - loss: 0.9573 - accuracy: 0.7195 - val_loss: 0.6945 - val_accuracy: 0.7964
    Epoch 7/25
    187/187 [==============================] - 55s 293ms/step - loss: 0.9472 - accuracy: 0.7213 - val_loss: 0.7157 - val_accuracy: 0.7933
    Epoch 8/25
    187/187 [==============================] - 57s 306ms/step - loss: 0.9525 - accuracy: 0.7196 - val_loss: 0.7164 - val_accuracy: 0.7983
    Epoch 9/25
    187/187 [==============================] - 56s 297ms/step - loss: 0.9479 - accuracy: 0.7274 - val_loss: 0.7623 - val_accuracy: 0.7871
    Epoch 10/25
    187/187 [==============================] - 56s 297ms/step - loss: 0.9270 - accuracy: 0.7279 - val_loss: 0.6931 - val_accuracy: 0.8007
    Epoch 11/25
    187/187 [==============================] - 54s 286ms/step - loss: 0.8885 - accuracy: 0.7356 - val_loss: 0.7967 - val_accuracy: 0.7868
    Epoch 12/25
    187/187 [==============================] - 55s 292ms/step - loss: 0.9245 - accuracy: 0.7350 - val_loss: 0.7469 - val_accuracy: 0.7968
    Epoch 13/25
    187/187 [==============================] - 51s 272ms/step - loss: 0.8948 - accuracy: 0.7418 - val_loss: 0.7850 - val_accuracy: 0.7902
    Epoch 14/25
    187/187 [==============================] - 51s 271ms/step - loss: 0.8829 - accuracy: 0.7441 - val_loss: 0.7671 - val_accuracy: 0.7944
    Epoch 15/25
    187/187 [==============================] - 51s 270ms/step - loss: 0.8895 - accuracy: 0.7411 - val_loss: 0.8210 - val_accuracy: 0.7842
    Epoch 16/25
    187/187 [==============================] - 50s 270ms/step - loss: 0.9051 - accuracy: 0.7377 - val_loss: 0.9133 - val_accuracy: 0.7698
    Epoch 17/25
    187/187 [==============================] - 50s 269ms/step - loss: 0.8964 - accuracy: 0.7409 - val_loss: 0.7984 - val_accuracy: 0.7878
    Epoch 18/25
    187/187 [==============================] - 50s 268ms/step - loss: 0.8880 - accuracy: 0.7436 - val_loss: 0.8293 - val_accuracy: 0.7924
    Epoch 19/25
    187/187 [==============================] - 50s 268ms/step - loss: 0.9058 - accuracy: 0.7409 - val_loss: 0.9220 - val_accuracy: 0.7675
    Epoch 20/25
    187/187 [==============================] - 50s 268ms/step - loss: 0.8772 - accuracy: 0.7461 - val_loss: 0.8364 - val_accuracy: 0.7851
    Epoch 21/25
    187/187 [==============================] - 51s 270ms/step - loss: 0.8937 - accuracy: 0.7487 - val_loss: 0.8493 - val_accuracy: 0.7854
    Epoch 22/25
    187/187 [==============================] - 50s 269ms/step - loss: 0.8969 - accuracy: 0.7442 - val_loss: 0.8614 - val_accuracy: 0.7826
    Epoch 23/25
    187/187 [==============================] - 49s 264ms/step - loss: 0.8597 - accuracy: 0.7530 - val_loss: 0.9160 - val_accuracy: 0.7845
    Epoch 24/25
    187/187 [==============================] - 51s 270ms/step - loss: 0.9021 - accuracy: 0.7512 - val_loss: 0.9065 - val_accuracy: 0.7835
    Epoch 25/25
    187/187 [==============================] - 51s 273ms/step - loss: 0.8675 - accuracy: 0.7574 - val_loss: 0.8605 - val_accuracy: 0.7916



![png](/assets/images/contents/keras-examples/Image classification via fine-tuning with EfficientNet/Image classification via fine-tuning with EfficientNet_33_1.png)

사전 학습된 가중치를 사용하기 전보다 훨씬 높은 정확도에서 학습을 시작할 수 있다.<br>



### 사전학습 가중치를 활용한 전이학습 - fine tuning

fine tuning은 레이어의 동결을 해제하고, 더 작은 학습률로 모델을 학습한다.<br>
이번 예제에서는 모든 레이어의 동결을 해제하지만, 데이터 세트에 따라 일부만 동결해제하는 것이 바람질할 수 있다.

ImageNet과 다른 데이터 세트를 사용할때, feature extractor도 조정해야 하기 때문에, 이 미세조정 단계가 더욱 중요하다.


```python
def unfreeze_model(model):
  for layer in model.layers[-20:]:
    if not isinstance(layer, layers.BatchNormalization):                         # BarchNormalization은 업데이트 X
      layer.trainable = True

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
  model.compile(
      optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
  )

unfreeze_model(model)

epochs=10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test, verbose=2)
plot_hist(hist)
```

    Epoch 1/10
    187/187 - 62s - loss: 0.5185 - accuracy: 0.8370 - val_loss: 0.7522 - val_accuracy: 0.8117 - 62s/epoch - 333ms/step
    Epoch 2/10
    187/187 - 52s - loss: 0.4729 - accuracy: 0.8511 - val_loss: 0.7391 - val_accuracy: 0.8170 - 52s/epoch - 277ms/step
    Epoch 3/10
    187/187 - 52s - loss: 0.4348 - accuracy: 0.8597 - val_loss: 0.7341 - val_accuracy: 0.8176 - 52s/epoch - 278ms/step
    Epoch 4/10
    187/187 - 52s - loss: 0.3972 - accuracy: 0.8712 - val_loss: 0.7538 - val_accuracy: 0.8141 - 52s/epoch - 279ms/step
    Epoch 5/10
    187/187 - 52s - loss: 0.3761 - accuracy: 0.8771 - val_loss: 0.7444 - val_accuracy: 0.8115 - 52s/epoch - 278ms/step
    Epoch 6/10
    187/187 - 52s - loss: 0.3450 - accuracy: 0.8878 - val_loss: 0.7707 - val_accuracy: 0.8095 - 52s/epoch - 279ms/step
    Epoch 7/10
    187/187 - 52s - loss: 0.3354 - accuracy: 0.8883 - val_loss: 0.7643 - val_accuracy: 0.8128 - 52s/epoch - 280ms/step
    Epoch 8/10
    187/187 - 53s - loss: 0.2918 - accuracy: 0.9046 - val_loss: 0.7674 - val_accuracy: 0.8130 - 53s/epoch - 281ms/step
    Epoch 9/10
    187/187 - 52s - loss: 0.2878 - accuracy: 0.9058 - val_loss: 0.7705 - val_accuracy: 0.8104 - 52s/epoch - 280ms/step
    Epoch 10/10
    187/187 - 54s - loss: 0.2725 - accuracy: 0.9068 - val_loss: 0.7731 - val_accuracy: 0.8116 - 54s/epoch - 288ms/step



![png](/assets/images/contents/keras-examples/Image classification via fine-tuning with EfficientNet/Image classification via fine-tuning with EfficientNet_36_1.png)


## EfficientNet 미세조정을 위한 팁

### on unfreezing layers:

- BatchNormalization 레이어는 계속 동결시킬 필요가 있다. 만약 훈련가능한 상태로 전환한다면, 첫번째 에포크의 정확도를 크게 떨어뜨릴 것이다.
- 어떤 경우엔, 모든 레이어를 동결해제 하는 대신, 일부 레이어만 동결해제하는 것이 유익할 수 있다. B7과 같은 더 큰 모델에서 미세조정을 더 빠르게 할 수 있다.
- 각 블록은 모두 끄거나 켜야한다.(블록 단위로 레이어를 동결하거나 학습시켜야함) 이는 아케텍처가 첫번째 레이어에서 마지막 레이어까지의 shortcut을 포함하기 때문이다.


### Some other tips for utilizing EfficientNet:

-  RMSprop를 사용하지마라, momentum과 learnin rate가 너무 높기 때문이다. 이는 사전학습된 가중치를 쉽게 손상시킨다.
- 배치크기가 작으면 정규화에 효과적이므로 검증 정확도가 향상된다.
- EfficinetNet의 큰 변형이 성능의 향상을 보장하지 않는다. 특히, 적은 데이터 또는 클래스를 가진 데이터 세트일수록 그렇다. EfficientNet이 크게 변형될 경우, 하이퍼파라미터 조정이 더욱 힘들어진다.


## Include_top 옵션비교

![png](/assets/images/contents/keras-examples/Image classification via fine-tuning with EfficientNet/include_top.PNG)
