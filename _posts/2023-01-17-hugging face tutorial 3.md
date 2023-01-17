---
layout: post
title: hugging face tutorials 3
subtitle: Preprocess
author: san9hyun
categories: sundries
banner : /assets/images/banners/squirrel.jpg
tags: huggingface tutorial 번역
---
## Preprocess

모델을 학습시키기 전에, 데이터를 모델에 입력하기 알맞은 형태로 변환해야한다.<br>
데이터가 텍스트, 이미지, 오디오 등 무엇이든지 텐서 배치로 변환되어야 한다.<br>
Transformers는 모델을 위한 데이터를 준비하는데 도움이 되는 전처리 클래스를 제공한다.<br>
이 튜토리얼에서는 다음을 배운다.<br>

- 토크나이저를 사용해 텍스트를 토큰으로 이루어진 시퀀스로 변환한다. 그리고 숫자로 표현(변환)하고 텐서로 변환한다.
- 이미지는 이미지프로세서를 이용해 텐서로 변환한다.
- 음성데이터는 Feature extractor(특징 추출기)를 사용해 오디오 파형에서 순차적 특징을 추출하고 텐서로 변환한다.(생략)
- 다중모드 입력은 프로세서를 사용하여 토크나이저와 특징 추출기 또는 이미지 프로세서를 결합한다.(생략)

## Natural Language Processing

토크나이저는 텍스트 데이터를 처리하는데 주요한 도구다.<br>
토크나이저는 정해진 규칙에 따라 텍스트를 토큰으로 분리(split)한다.<br>
그 토큰들은 숫자로 변환되고 텐서로 변환된다. 그리고 모델에 입력된다. <br>

사전학습 모델을 사용할 때는, 연관된 사전학습 토크나이저를 사용하는 것이 중요하다.<br>
그래야 텍스트가 사전 훈련 말뭉치(corpus)와 동일한 방식으로 분할되고, 사전 훈련 중에 동일한 vocab을 사용한다.

AutoTokenizer.from_pretrained() 메서드로 사전학습된 토크나이저를 로드하면, 사전 훈련된 어휘가 다운로드 된다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Do not meddle in the affairs of wizards, for they are subtle and quick to anger.")
print(encoded_input)
{'input_ids': [101, 2079, 2025, 19960, 10362, 1999, 1996, 3821, 1997, 16657, 1010, 2005, 2027, 2024, 11259, 1998, 4248, 2000, 4963, 1012, 102],
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```
토크나이저는 3가지 중요한 아이템 사전을 리턴한다.

- input_ids는 문장의 각 토큰에 대응하는 인덱스이다.
- attention_mask는 토큰이 attended 되어야할지 여부를 나타낸다.
- token_type_ids는 둘 이상의 시퀀스가 있을 때, 토큰이 속한 시퀀스를 식별한다.

input_ids를 디코딩하여 입력을 반환할 수 있다.
```python
tokenizer.decode(encoded_input["input_ids"])
'[CLS] Do not meddle in the affairs of wizards, for they are subtle and quick to anger. [SEP]'
```
[CLS], [SEP]를 토크나이저가 추가한 것을 알 수 있다.<br>
모든 모델에 위와같은 특수토큰이 필요한 것은 아니지만, 필요한 경우 토크나이저가 자동으로 토큰을 추가한다.<br>

전처리할 문장이 여러개인 경우 토크나이저에 list로 전달한다.<br>

```python
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_inputs = tokenizer(batch_sentences)
print(encoded_inputs)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1]]}
```
 ### pad  

문장은 항상 같은 길이가 아니다. 모델에 입력되는 텐서가 일정한 shape을 가져야하기 때문에, 길이가 다른것은 문제가 될 수 있다.<br>
패딩(Padding)은 짧은 문장에 특별한 패딩 토큰을 추가해 텐서가 직사각형이 되도록 하는 전략이다.

padding 파라미터를 True로 설정하면, 가장 긴 시퀀스와 일치하도록 배치에서 패딩을 수행한다.<br>

```python
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True)
print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0], 
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102], 
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```
### truncation

시퀀스가 때로는 너무 길 수 있다. 이 경우 시퀀스를 잘라야한다.<br>
truncaiton = True로 설정하면 허용하는 최대길이를 넘지않도록 시퀀스를 자를 수 있다.<br>

```python
batch_sentences = [
  "But what about second breakfast?",
  "Don't think he knows about second breakfast, Pip.",
  "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True)
print(encoded_input)
{'input_ids': [[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
               [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
               [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]],
 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]}
```

### build tensor

마지막으로, 토크나이저가 tensor를 리턴하도록 하자.<br>
return_tensors 파라미터를 pt 또는 tf로 세팅하면 된다. pt:pytorch, tf:tensorflow

```python
batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
]
encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
print(encoded_input)
{'input_ids': tensor([[101, 1252, 1184, 1164, 1248, 6462, 136, 102, 0, 0, 0, 0, 0, 0, 0],
                      [101, 1790, 112, 189, 1341, 1119, 3520, 1164, 1248, 6462, 117, 21902, 1643, 119, 102],
                      [101, 1327, 1164, 5450, 23434, 136, 102, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}
```

## Computer vison

비전 태스크에서는 이미지 프로세서가 필요하다. 이미지 프로세서는 이미지를 처리하고 텐서로 변환하도록 설계되어있다.<br>
```python
from datasets import load_dataset

dataset = load_dataset("food101", split="train[:100]")
```

Datasets의 Image기능을 사용해 이미지를 살펴보자.<br>
```python
dataset[0]["image"]
```


![png](/assets/images/contents/huggingface/sample.png)


AutoImageProcessor.from_pretrained()로 이미지 프로세서를 로드할 수 있다.<br>
```python
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```
비전 태스크에서는, 일부 유형의 데이터를 증강하는 것이 흔한 처리 방법이다.<br>
데이터 증강을 위한 여러 라이브러리가 있지만, 이 튜토리얼에서는 torchvision의 transforms 모듈을 사용한다.<br>

1.이미지 프로세서로 이미지를 정규화하고, Compose를 사용해 RandomResizedCrop과 ColorJitter를 연결해서 변환한다.<br>

```python
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
    
_transforms = Compose([RandomResizedCrop(size), ColorJitter(brightness=0.5, hue=0.5), ToTensor(), normalize])
```

2.모델은 입력값으로 pixel_values를 받는다. pixel_values를 생성하는 함수를 만들자.<br>

```python
def transforms(examples):
  examples["pixel_values"] = [_transforms(image.convert("RGB")) for image in examples["image"]]
  return examples
```

3.Datasets의 set_transform을 사용하면 transforms 함수를 적용할 수 있다.

```python
dataset.set_transform(transforms)
```
4.이제 이미지에 접근하면, pixel_value가 추가되었음을 알 수 있다.

```python
dataset[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=384x512 at 0x7F1A7B0630D0>,
 'label': 6,
 'pixel_values': tensor([[[ 0.0353,  0.0745,  0.1216,  ..., -0.9922, -0.9922, -0.9922],
          [-0.0196,  0.0667,  0.1294,  ..., -0.9765, -0.9843, -0.9922],
          [ 0.0196,  0.0824,  0.1137,  ..., -0.9765, -0.9686, -0.8667],
          ...,
          [ 0.0275,  0.0745,  0.0510,  ..., -0.1137, -0.1216, -0.0824],
          [ 0.0667,  0.0824,  0.0667,  ..., -0.0588, -0.0745, -0.0980],
          [ 0.0353,  0.0353,  0.0431,  ..., -0.0039, -0.0039, -0.0588]],
 
         [[ 0.2078,  0.2471,  0.2863,  ..., -0.9451, -0.9373, -0.9451],
          [ 0.1608,  0.2471,  0.3098,  ..., -0.9373, -0.9451, -0.9373],
          [ 0.2078,  0.2706,  0.3020,  ..., -0.9608, -0.9373, -0.8275],
          ...,
          [-0.0353,  0.0118, -0.0039,  ..., -0.2392, -0.2471, -0.2078],
          [ 0.0196,  0.0353,  0.0196,  ..., -0.1843, -0.2000, -0.2235],
          [-0.0118, -0.0039, -0.0039,  ..., -0.0980, -0.0980, -0.1529]],
 
         [[ 0.3961,  0.4431,  0.4980,  ..., -0.9216, -0.9137, -0.9216],
          [ 0.3569,  0.4510,  0.5216,  ..., -0.9059, -0.9137, -0.9137],
          [ 0.4118,  0.4745,  0.5216,  ..., -0.9137, -0.8902, -0.7804],
          ...,
          [-0.2314, -0.1922, -0.2078,  ..., -0.4196, -0.4275, -0.3882],
          [-0.1843, -0.1686, -0.2000,  ..., -0.3647, -0.3804, -0.4039],
          [-0.1922, -0.1922, -0.1922,  ..., -0.2941, -0.2863, -0.3412]]])}
```
변환이 적용된 이미지는 다음과 같이 무작위로 잘렸으며, 색상이 다르다.<br>
```python
import numpy as np
import matplotlib.pyplot as plt

img = dataset[0]["pixel_values"]
plt.imshow(img.permute(1, 2, 0))
```

![png](/assets/images/contents/huggingface/sample_transformed.PNG)
