---
layout: post
title: hugging face tutorials 1 
subtitle: pipelines for inference
author: san9hyun
categories: code-example
banner : /assets/images/banners/squirrel.jpg
tags: huggingface tutorial 번역
---

## 1.들어가기 전에
hugging face는 사전학습 모델들과, 토크나이저들을 쉽게 불러와서 사용할 수 있다.<br>
실제로 roberta 모델을 문서 분류 태스크에 사용해봤었는데 그 기능은 정말 강력했다....<br>
코드 몇줄이면 여러가지 task에 대해 추론까지 할 수 있다..

hugging face를 더 잘, 제대로 사용하기 위해서 시간 날때 Docs를 꼼곰하게 읽어보려고한다.<br>

## 2.추론을 위한 파이프라인

pipeline().을 사용하면 모든 언어, 비전, 음성 그리고 다중모드 작업에서 [transformer의 모델들을](https://huggingface.co/models) 간단하게 사용할 수 있다.
특정 양식에 대한 경험이 없고, 모델의 코드에 익숙하지 않아도 쉽게 추론에 사용할 수 있다.<br>
이 튜토리얼에서는  다음을 알려준다.

- 추론을 위한 pipeline 사용
- 특정 토크나이저 또는 모델 사용
- 오디오 비전 및 다중 모드 작업에 파이프라인 사용

## 2-1파이프라인 사용

각 태스크에는 연결된 pipeline()이 있지만, 모든 태스크별 파이프라인을 포함하는 pipeline 추상화를 사용하는 것이 더 간단하다.
그 pipline().은 태스트 추론을 위한 기본모델과 전처리 클래스를 자동으로 로드한다.<br>

pipline().을 생성하고 추론 작업을 지정한다.<br>
```python
from transformers import pipeline
generator = pipeline(task="text-generation")
```

입력 텍스트를 pipline().에 전달한다.<br>
```python
generator("Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone")
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Iron-priests at the door to the east, and thirteen for the Lord Kings at the end of the mountain'}]
```
만약 입력 텍스트가 둘 이상이라면, list 형식으로 전달한다.<br>
```python
generator(
    [
        "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
        "Nine for Mortal Men, doomed to die, One for the Dark Lord on his dark throne",
    ]
) 
```
매개변수 또한 존재하는데, 예를들어 둘 이상의 출력을 생성하려면, 아래와 같이 num_return_sequences 매개변수로 설정할 수 있다.<br>
```python
generator(
    "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone",
    num_return_sequences=2,
)  
```
## 2-2모델, 토크나이저 선택

pipeline().은 허브의 모든 모델(트랜스포머 제공 모델)들을 받을 수 있다. 적절한 모델을 선택하면 해당하는 AutoModelFor와 AutoTokenizer 클래스를 로드한다.
예를 들어, causal language 모델링 태스크를 위해 AutoModelForCasualLM 클래스를 로드한다.<br>
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
```
태스크를 위한 pipline().을 생성하고, 로드했던 모델과 토크나이저를 지정한다.<br>
```python
from transformers import pipleine
generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
```
입력 텍스트를 pipline().에 전달하여 텍스트를 생성한다.
```python
generator(
    "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone"
)  
[{'generated_text': 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Seven for the Dragon-lords (for them to rule in a world ruled by their rulers, and all who live within the realm'}]
```
## 2-3오디오 파이프라인
pipline().은 오디오 분류와 자동 음성 인식과 같은 오디오 태스크도 지원한다.

예를들어, 이 오디오 클립애 대해 감정분류를 해보자.<br>
```python
from datasets import load_dataset
import torch

torch.manual_seed(42)
ds = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
audio_file = ds[0]["audio"]["path"]
```
audio classification 모델을 허브에서 찾아 pipeline().에 지정해준다.
```python
audio_classifier = pipeline(
    task="audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)
```
오디오 파일을 pipeline()에 전달해준다.
```python
preds = audio_classifier(audio_file)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds
[{'score': 0.1315, 'label': 'calm'}, {'score': 0.1307, 'label': 'neutral'}, {'score': 0.1274, 'label': 'sad'}, {'score': 0.1261, 'label': 'fearful'}, {'score': 0.1242, 'label': 'happy'}]
```
## 2-4비전 파이프라인
비전 태스크에도 동일하게 pipline()응 사용한다.<br>
태스크를 지정하고 이미지를 분류기에 전달한다.이미지는 링크 또는 로컬 경로 일 수도 있다.

```python
vision_classifier = pipeline(task="image-classification")

preds = vision_classifier(
  images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
preds
[{'score': 0.4335, 'label': 'lynx, catamount'}, {'score': 0.0348, 'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'}, {'score': 0.0324, 'label': 'snow leopard, ounce, Panthera uncia'}, {'score': 0.0239, 'label': 'Egyptian cat'}, {'score': 0.0229, 'label': 'tiger cat'}]
```
위 코드를 보면, pipline에 model을 설정해주지 않았다.<br>
model을 설정하지 않으면, 디폴트로 특정 모델을 사용하게 되는데, 이러한 방법(model 설정 x)은 추천하지 않는다고 한다.<br>
실제 위 코드를 실행시키면 아래와 같은 경고를 받게 된다.

```text
No model was supplied, defaulted to google/vit-base-patch16-224 and revision 5dca96d (https://huggingface.co/google/vit-base-patch16-224).
Using a pipeline without specifying a model name and revision in production is not recommended.
```

## 2-5다중 모드 파이프 라인
pipeline().은 둘 이상의 양식을 지원한다. 예를 들어, 시각적 질의 응답 태스크는 텍스트와 이미지를 결합한다.<br>

예를 들어보자,
```python
image = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
question = "Where is the cat?"
```
vqa 파이프라인을 만들고 이미지와 질문을 전달한다.<br>

```python
vqa = pipline(task="vqa")
preds = vqa(image=image, question=question)
preds 
[{'score': 0.44030314683914185, 'label': 'lynx, catamount'},
 {'score': 0.034334052354097366,
  'label': 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor'},
 {'score': 0.03214803338050842,
  'label': 'snow leopard, ounce, Panthera uncia'},
 {'score': 0.02353905513882637, 'label': 'Egyptian cat'},
 {'score': 0.02303415723145008, 'label': 'tiger cat'}]
```
자연어, 비전, 다중태스크까지 추론과정이 코드 몇줄이면 끝나다. <br>
정말 굉장하다.
