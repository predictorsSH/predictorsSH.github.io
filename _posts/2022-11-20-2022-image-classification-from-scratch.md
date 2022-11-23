---
layout: post
title: image classification from scratch
subtitle: keras example - Classification of images of dogs and cats
author: san9hyun
categories: keras-example
banner : /assets/images/banners/solar-system.jpg
tags: Xception ResidualNetwork
---
**dark 모드로 보기** <br>
[keras code example](https://keras.io/examples/vision/image_classification_from_scratch/) 을 따라 공부한 것 입니다.

## Description

고양이와 개 이미지를 분류하는 모델 학습

## Setup


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

## Load the data

-이미지 ZIP 파일 다운로드 <br>


```python
!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  786M  100  786M    0     0   180M      0  0:00:04  0:00:04 --:--:--  183M



```python
# 압축해제
!unzip -q kagglecatsanddogs_5340.zip
```


```python
!ls PetImages
```

    Cat  Dog


### 손상된 이미지 필터링

헤더에 "JFIF" 문자열이 없는 잘못된 인코딩 이미지를 필터링


```python
import os
```


```python
num_skipped = 0

for folder_name in ("Cat","Dog"):
  folder_path = os.path.join("PetImages",folder_name) #folder_path -> "PetImages/Cat" , "PetImages/Dog"
  for fname in os.listdir(folder_path):
    fpath = os.path.join(folder_path, fname)
    try:
      fobj = open(fpath, "rb")
      is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
    finally:
      fobj.close()

    if not is_jfif:
      num_skipped += 1
      os.remove(fpath)

print("Delete %d images" % num_skipped)
```

    Delete 1590 images


## 데이터 셋 생성


```python
image_size = (180,180)
batch_size = 128

#학습 데이터
train_ds= tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size = image_size,
    batch_size = batch_size,
)

#검증데이터
val_ds= tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size = image_size,
    batch_size = batch_size,
)
```

    Found 23410 files belonging to 2 classes.
    Using 18728 files for training.
    Found 23410 files belonging to 2 classes.
    Using 4682 files for validation.


## 데이터 시각화


```python
import matplotlib.pyplot as plt
```


```python
plt.figure(figsize=(5,5))

#take(n)에서 n은 batch 개수를 의미, 즉 train_ds.take(1)은 train_ds에서 1 batch(128개의 데이터)만 불러오라는 의미
for images, labels in train_ds.take(1):
  for i in range(4):
    if labels[i]==1:
      title = "dog"
    else :
      title= "cat"
    ax = plt.subplot(2,2,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(title)
    plt.axis("off")
```


![png](/assets/images/contents/keras-examples/iamge_classification_from_scratch/image_classification_from_scratch_17_0.png)


## 데이터증강(Data Augmentation)


```python
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
    ]
)
```


```python
plt.figure(figsize=(10,10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3,3,i+1)
    plt.imshow(augmented_images[0].numpy().astype("uint8")) # 마지막 이미지만 시각화, 마지막 이미지 shape이 (128,180,180,3) 이기 떄문에 (180,180,3)으로 변경
    plt.axis("off")
```


![png](/assets/images/contents/keras-examples/iamge_classification_from_scratch/image_classification_from_scratch_20_0.png)


### 데이터 표준화(Standardizing the data)

RGB channel 값이 0에서 255사이의 값을 가지므로, 0에서 1사이값을 가지도록 변환<br>


## 데이터 전처리


두가지 방법이 존재
- 모델내에서 처리 : GPU를 사용할 수 있다는 장점이 있음. 데이터 증강은 testime에 사용 하지 않음
- 데이터 셋에 직접 적용 : CPU로 학습을 할때 더 효율적. 어떤 옵션을 선택할지 모를때 이 옵션을 선택하자


```python
#  모델 내에서 처리

# inputs = keras.Input(shape=input_shape)
# x = data_augmentation(inputs)
# x = layers.Rescaling(1./255)(x)  #데이터 표준화
```


```python
#  데이터 셋에 직접 적용
#  augmented_train_ds = train_ds.map(lambda x, y:(data_augmentation(x, training=True),y))
```

## 데이터셋 구성


```python
train_ds = train_ds.map(
    lambda img, label: (data_augmentation(img), label),
    num_parallel_calls=tf.data.AUTOTUNE,
)

#현재 배치가 처리되는 동안 다음 배치를 준비할 수 있음. (현재 배치가 학습되는 동안 이후 배치 데이터를 메모리에 올리는 건가?)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
```

## 모델 빌드

간단한 Xeption 모델 빌드


```python
from tensorflow.python.util.tf_export import KERAS_API_NAME
def make_model(input_shape, num_classes):
  inputs = keras.Input(shape=input_shape)

  x = layers.Rescaling(1.0/255)(inputs)
  x = layers.Conv2D(128,3,strides=2, padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  previous_block_activation = x # set aside residual

  for size in [256,512,728]:
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)


    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    #잔차학습
    residual = layers.Conv2D(size, 1, strides=2, padding="same")(
        previous_block_activation
    )
    x = layers.add([x, residual]) # x, residual 원소합 연산 수행
    previous_block_activation = x

  x = layers.SeparableConv2D(1024,3,padding="same")(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation("relu")(x)

  x = layers.GlobalAveragePooling2D()(x)

  if num_classes == 2:
    activation = "sigmoid"
    units = 1
  else :
    activation = "softmax"
    units = num_classes

  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(units, activation=activation)(x)
  return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)
keras.utils.plot_model(model, show_shapes=True)

```




![png](/assets/images/contents/keras-examples/iamge_classification_from_scratch/image_classification_from_scratch_31_0.png)



## 모델 학습


```python
epochs = 2

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")
]

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"],
              )

model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds
)
```

    Epoch 1/2
    147/147 [==============================] - 329s 2s/step - loss: 0.4348 - accuracy: 0.8007 - val_loss: 0.9524 - val_accuracy: 0.4957
    Epoch 2/2
    147/147 [==============================] - 331s 2s/step - loss: 0.3293 - accuracy: 0.8584 - val_loss: 0.8550 - val_accuracy: 0.4981





    <keras.callbacks.History at 0x7f8f721c9a50>



## 새로운 데이터 추론


```python
img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = float(predictions[0])
print(f"This image is {100 * (1 - score):.2f}% cat and {100 * score:.2f}% dog.")
```

    1/1 [==============================] - 1s 574ms/step
    This image is 79.28% cat and 20.72% dog.
