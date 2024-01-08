---
layout: post
title: hugging face tutorials 2
subtitle: Load pretrained instances with an AutoClass
author: san9hyun
categories: code-example
banner : /assets/images/banners/squirrel.jpg
tags: huggingface tutorial 번역
---

## 사전 학습된 인스턴스 로드
라이브러리를 쉽고, 간단하게 그리고 유연하게 사용하도록 하기 위한 Transformers의 핵심 철학의 일환으로
AutoClass는 주어진 체크포인트에 적절한 아키텍처를 자동으로 유츄하고 로드한다. <br>
<br>
from_pretrained() 방법을 사용하면 모든 아키텍처에 대해
사전 훈련된 모델을 빠르게 로드 할 수 있으므로 훈련하는데 리소르를 할애할 필요가 없다.<br>
<br>
여기서 아키텍처는 모델의 골격을 의미하고, 체크포인트는 주어진 아케텍처의 가중치를 의미한다.<br>


이 튜토리얼에서는 아래와 같은 내용을 배운다.

- 사전 학습된 토크나이저를 로드합니다.
- 사전 훈련된 이미지 프로세서 불러오기
- 사전 훈련된 특징 추출기를 불러옵니다.
- 사전 훈련된 프로세서를 로드합니다.
- 사전 학습된 모델을 로드합니다.

## AutoTokenizer

거의 모든 NLP 태스크는 토크나이저와 함께 시작한다. 토크나이저는 입력 데이터를 모델이 처리할 수 있는 형식으로 변환한다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```
```python
sequence = "In a hole in the ground there lived a hobbit."
print(tokenizer(sequence))
{'input_ids': [101, 1999, 1037, 4920, 1999, 1996, 2598, 2045, 2973, 1037, 7570, 10322, 4183, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

## AutoImageProcessor

비전 태스크에서, 이미지 프로세서가 이미지를 올바른 입력 형식으로 처리한다.

```python
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
```
##  AutoFeatureExtractor

오디오 작업일 경우, AutoFeatureExtractor.from_pretrained()를 사용하여 오디오 신호를 올바른 입력 형식으로 처리한다.

```python
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
```

## AutoProcessor

다중 모드 작업에는 두가지 유형의 전처리 도구를 결합한 프로세서가 필요하다.<br>
예를 들어, LayoutLMV2 모델은 이미지 처리를 위한 프로세서와, 텍스트를 처리하기 위한 토크나이저가 필요하다.<br>

AutoProcessor.from_pretrained()를 사용하여, 토크나이저와 이미지 처리 프로세서를 결합하는 프로세서를 로드할 수 있다.

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
```

## AutoModel

마지막으로, AutoModelFor 클래스를 사용하면, 주어진 작업에 대해 사전 훈련된 모델을 로드할 수 있다.<br>
예를 들어,  AutoModelForSequenceClassification.from_pretrained() 를 사용하여 시퀀스 분류를 위한 모델을 로드한다.

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
```
