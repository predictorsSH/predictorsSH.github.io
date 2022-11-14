---
layout: post
title: Towards Total Recall in Industrial Anomaly Detection(작성중)
subtitle: PatchCore paper
author: san9hyun
categories: paper
banner : /assets/images/banners/book.jpg
tags: DataScience paper patchcore
---

## Towards Total Recall in Industrial Anomaly Detection

** 초안 작성 하며 공부중, 추후 가다듬을 예정

## [[paper]](https://arxiv.org/abs/2106.08265)


## Abstract

이 연구의 도전 과제는, 정상데이터만을 사용하여 모델을 학습시키는 cold-start 문제이다. *cold-start(데이터가 없어 적절한 모델을 만들기 어려운 문제) <br>
문제 해결을 위한 가장 좋은 접근법은, ImageNet으로 데이터를 임베딩하고 outlier detection 모델로 이상치를 탐지 하는 것이다.<br>

이 논문에서 우리는 PatchCore를 제안한다.<br>
PatchCore는 탐지와, localization 에서 최첨단 성능을 달성하면서, 빠른 추론시간을 제공한다. *Localization(object가 이미지안의 어디 위치에 있는지 알려주는 것) <br>
PatchCore는 image-level anomaly detection AUROC 점수를 99.6%까지 달성한다.

## Indroduction
인간의 적은수의 정상데이터만 보고도, 기대되는 분산과 이상치를  구별할 수 있다. <br>
이 연구는 이것의 컴퓨터 버젼인, 산업용 이미지를 검사하는 cold-start 이상탐지 문제를 다룬다.<br>
*산업용 이미지 검사 이상탐지 문제 : 이미지를 보고 부품의 결함을 탐지하는 문제

기존 cold-start 문제 연구들은, 오토인코딩, GANs 와 같은 모델들로, 정상적인 분포를 학습하는 것에 의존한다.
최근에는, target distribution 대한 adaptation 없이(target에 대한 학습 없이?), ImageNet 분류모델로부터 common deep representations 활용하는 것이 제안된다.<br>
<br>
adaptation 생략하고도, 이러한 모델들은 결함의 localization 이상 탐지에 강한 성능을 보여준다.<br>
이런 테크닉의 핵심은 deep feature representations의 multiscale nature를 이용하면서, 테스트 샘플과 정상 샘플간의 특징을 대조하는 것이다.<br>
<br>
그런데 미묘한 결함의 분할,추출(segmentation)은  높은 해상도(high resolution, 입력층에 가까움) feature로 커버가 되는 반면, 
구조적 편차와 전체 이미지 수준의 Anomaly Detection은 더 높은 추상화 feature(입력층으로부터 멈)에 의해 커버 된다.<br>
<br>
따라서, 이 접근법의 본질적인 단점은, target에 대한 adaption이 없기 때문에(target 데이터를 학습 하지 않기 때문에) 높은 추상화 수준에서 특징을 대조하는 것에 대한 신뢰도가 제한적이라는 것이다.<br>
(ImageNet의 높은 추상 특징은 산업환경에서 요구되는 추상 특징과 매우 다름.)

이 논문에서는 아래 3가지 효과적인 해결방법으로써 PatchCore를 제안한다. <br>
1. maximizing nominal information available at test time. 테스트시 정상 정보를 가능한 최대화.
2. reducing biases towards ImageNet classes. ImageNet의 클래스로부터의 편향을 감소.
3. retaining high inference speeds. 높은 추론 속도 유지

PatchCore는 mid-level network feature를 사용하여, ImageNet classes에 대한 편향을 최소화 하면서도,
이웃 지역에 대한 feature aggregation은 충분히 공간적 문맥 정보를 보장한다.
결과적으로, PatcCore는 광범위한 메모리 뱅크를 통해 테스트시 사용 가능한 정상 정보를 최적으로 활용할수 있다.<br>
<br>
마지막으로, PatchCore는 추출된 패치 레벨 메모리 뱅크의 중복성을 줄이고, 스토리지 메모리와 추론 시간 단축을 위해 탐욕적 coreset 서브샘플링을 추가로 도입하였다.
따라서 PatcCore는 매우 실용적이고, 산업에 활용하기 매력적이다.

## Related Works

PatchCore에서 사용한 특정 컴포넌트들은 SPADE와 PaDiM과 관련이 있다.<br>
SPADE는 정상 피처의 memory-bank를 사용하며, 정상 피처는 사전 학습된 모델에서 추출 된다. 그리고 pixel-level과 image-level의 이상탐지에 대해 분리된 접근을 한다.<br>
PatchCore또한 유사하게 memory-bank를 사용하지만, neighbourhoodaware patch-level features를 사용하는 차이가 있다.<br>
그리고 memory-bank는 서브 샘플링된 coreset이다(정상 데이터를 전부 활용하지 않음). 이 방식은 더 높은 성능에 더 낮은 추론 비용을 보장한다.<br>
<br>
PatchCore의 image-level의 anomaly detection과 segmentation 두가지 모두에 대한 patch-level의 접근은 PaDiM과 연관이있다.<br>
PaDiM은 patch-level의 anomalt detection을, 각 patch의 고유한 마할라노비스 거리를 측정하는 것으로 제한하는 반면에, PatchCore는 모든 패치들에 대해 동일하게 접근 가능한
효율적인 patch-feature memory bank를 활용한다. (아직 무슨말인지 모르겠지만, 아래 Method를 보며 이해해보자!)

## Method

PatchCore는 아래 3가지 단계로 구성된다.<br>
- Locally aware patch features
- Coreset-reduced patch-feature memory bank
- Anomaly Detection with PatchCore

### Locally aware path features

PatchCore는 사전학습 모델 $\phi$ 을 사용하는데, 특정 네트워크 층의 피처가 중요한 역할을 한다.<br>
이미지 $x_i$에 대해, $\phi$의 j번째층 피처를 아래와 같이 표현할 수 있다.<br>

$$ \phi_{i,j} = \phi_j(x_i) $$

피처를 선택하는 한가지 방법으로, 마지막 층의 피처 표현을 사용할 수 있다.<br>
그러나 두가지 문제가 발생한다. 첫째, 정상데이터의 지역적 정보를 잃는다<br>
둘째,  ImageNet의 마지막 층 피처들은 이미지 분류 작업에 편향되어있으며, 이는 연구에서의 anomaly detection 작업과, 평가된 데이터 세트와는 많은 차이가 있다.<br>
따라서 PatchCore는 midel-level 피처 표현을 사용한다.<br>

patch 표현을 공식화 하기 위해서 이전에 소개한 표현법을 확장한다.<br>
피처맵 $ \phi_{i,j} $ 이  depth $c^*$, heigjt $h^*$ 그리고 width $w^*$ 세가지 차원을 가진 텐서라고 가정하자<br>
우리는, 특정 포지션(h,w)에서의 C차원 피처 slice를 아래와 같이 표현 할 수 있다.<br>

$$ \phi_{i,j}(h,w) = \phi_j(x_i,h,w) $$

이상적으로는, 각 patch-representation은 지역 공간 변화에 강한 의미가 이있는 anomalous context를 설명하기 위해 충분히 큰 receptive field size로 동작해야한다.<br>
이는 신경망의 더 깊은 층으로 내려가면 가능하지만, 그렇게 되면 ImageNet에 더 고유해져서 당면한 이상탐지 작업과는 덜 관련이 있는 반면, 교육 비용이 증가하고, feature map 해상도는 떨어진다.<br>

따라서, 공간 해상도를 잃지 않으면서, 작은 spatial deviations에 견고한 feature를 구성하기 위해 local neighbourhood aggregation 기법을 사용한다.<br>

neighbourhood feature vectors를 아래와 같이 표현하고,

$$ N_p^{(h,w)} = {(a,b)|a \in [h - [p/2], ...,h+[p/2]], b \in [w-[p/2], ..., w+[[/2]]]} $$ 


locally aware features 는 아래와 같이 표현할 수 있다.

$$ \phi_{i,j}(N_p^{(h,w)}) = f_{agg}({\phi_{i,j}(a,b)|(a,b) \in N_p^{(h,w)}}) $$
