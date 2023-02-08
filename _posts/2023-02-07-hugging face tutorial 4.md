---
layout: post
title: hugging face tutorials 4
subtitle: Preprocess
author: san9hyun
categories: sundries
banner : /assets/images/banners/squirrel.jpg
tags: huggingface tutorial 번역
---

```python
!pip install datasets
!pip install transformers
```

## Fine-tune a pretrained model
사전학습 모델을 사용하면 상당한 이점이 있다. 계산 비용과 탄소 발자국(carbonfootprint)을 줄이고, 처음부터 학습하지 않아도 state-of-the-art 모델들을 사용할수 있다. <br>

Transformers는 다양한 과제수행을 위한, 수천개의 사전학습 모델을 제공한다. 사전학습모델을 사용할때, 특정한 데이터세트로 사전학습모델을 학습한다. 이 학습은 매우 강력한 훈련 기술인 fine-tuning으로 알려져있다. <br>

이 튜토리얼에서 당신은 당신이 선택한 딥러닝 프레임워크로, fine-tuning을 진행할 것이다.

## Prepare a dataset

먼저, 훈련을 위한 데이터세트를 다운로드하자. <br>
사용할 데이터세트는 Yelp Reviews다.


```python
from datasets import load_dataset
dataset = load_dataset("yelp_review_full")
```



```python
dataset["train"][100]
```


    {'label': 0,
     'text': 'My expectations for McDonalds are t rarely high. But for one to still fail so spectacularly...that takes something special!\\nThe cashier took my friends\'s order, then promptly ignored me. I had to force myself in front of a cashier who opened his register to wait on the person BEHIND me. I waited over five minutes for a gigantic order that included precisely one kid\'s meal. After watching two people who ordered after me be handed their food, I asked where mine was. The manager started yelling at the cashiers for \\"serving off their orders\\" when they didn\'t have their food. But neither cashier was anywhere near those controls, and the manager was the one serving food to customers and clearing the boards.\\nThe manager was rude when giving me my order. She didn\'t make sure that I had everything ON MY RECEIPT, and never even had the decency to apologize that I felt I was getting poor service.\\nI\'ve eaten at various McDonalds restaurants for over 30 years. I\'ve worked at more than one location. I expect bad days, bad moods, and the occasional mistake. But I have yet to have a decent experience at this store. It will remain a place I avoid unless someone in my party needs to avoid illness from low blood sugar. Perhaps I should go back to the racially biased service of Steak n Shake instead!'}



텍스트를 처리하고 서로다른 텍스트 길이를 조정하기 위해서 토크나이저 필요하다 <br> Datasets map 메서드를 사용하면, 데이터세트를 한번에 전처리할 수 있다.



```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(example):
  return tokenizer(example["text"],padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```



만약 조금더 적은 데이터세트로 튜터리얼을 진행하고 싶으면 아래 코드를 실행하자


```python
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
```

## Train

### pytorch Trainer로 학습

Transformers는 훈련에 최적화된 Trainer 클래스를 제공한다. Trainer API는 넓은 범위의 훈련 옵션, 기울기 누적, 혼합 정밀도 등을 지원한다. <br>

모델을 로드하고, 기대되는 label 수를 지정한다. <br>
사전 훈련된 헤드는 폐기되고, 무작위로 초기화된 분류 헤드로 대체된다.<br>


```python
from  transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
```


    Downloading (…)"pytorch_model.bin";:   0%|          | 0.00/436M [00:00<?, ?B/s]


    Some weights of the model checkpoint at bert-base-cased were not used when initializing BertForSequenceClassification: ['cls.predictions.decoder.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.weight', 'cls.seq_relationship.weight', 'cls.predictions.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-cased and are newly initialized: ['classifier.weight', 'classifier.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


## Training hyperparameters

다음 단계로 TrainingArguments 클래스를 생성한다.<br>
TrainingArguments는 튜닝할 수 있는 모든 하이퍼파라미터가 포함되어있다.<br>


```python
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")
```

## Evaluate

Trainer는 훈련중에 모델 성능을 자동으로 평가하지 않는다.
메트릭을 확인하려면 Trainer에 함수를 전달해야한다.


```python
!pip install evaluate
```



```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
```


    Downloading builder script:   0%|          | 0.00/4.20k [00:00<?, ?B/s]


예측의 정확도를 측정하려면, metric.compute를 호출해라.
예측을 metric에 전달하기 전에, 예측을 logits으로 변환해야한다.(logits을 예측으로 변환 아닌가? 오류?)


```python
def compute_metrics(eval_pred):
  logits, labels = eval_pred # transformer model은 logits을 반환
  predictions = np.argmax(logits, axis=-1) #logits을 class로 변환
  return metric.compute(predictions=predictions, references=labels)
```

미세 조정중에 평가 측정항목을 모니터링하려면 evaluation_strategy= "epoch" 를 통해, 각 에포크가 끝날때 측정항목을 확인 할 수 있다.


```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
```

## Trainer

모델, training arguments, 학습, 검증 데이터셋, 그리고 평가로 Trainer 객체를 생성한다.


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
```

그리고 train()을 호출하면 fine-tuning을 수행한다.


```python
trainer.train()
```
![img.png](/assets/images/contents/code-examples/hugging face/trainer_train.PNG)
