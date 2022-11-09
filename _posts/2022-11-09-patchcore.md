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
미묘한 결함의 분할,추출(segmentation)은  높은 해상도(high resolution, 입력층에 가까움) feature로 커버가 되는 반면에, 
구조적 편차와 전체 이미지 수준의 Anomaly Detection은 더 높은 추상화 feature(입력층으로부터 멈)에 의해 커버 된다.<br>
<br>
이 접근법의 본질적인 단점은, target에 대한 adaption이 없기 때문에(target 데이터를 학습 하지 않기 때문에) 높은 추상화 수준에서 특징을 대조하는 것에 대한 신뢰도가 제한적이라는 것이다.<br>
(ImageNet의 높은 추상 특징은 산업환경에서 요구되는 추상 특징과 매우 다름.)

이 논문에서는 3가지 효과적인 방법으로 PatchCore를 제안한다. <br>
1. maximizing nominal information available at test time. 테스트시 정상 정보를 가능한 최대화.
2. reducing biases towards ImageNet classes. ImageNet의 클래스로부터의 편향을 감소.
3. retaining high inference speeds. 높은 추론 속도 유지

PatchCore는 mid-level network feature를 사용하여, ImageNet classes에 대한 편향을 최소화 하면서도,
이웃 지역에 대한 feature aggregation은 충분히 공간적 문맥 정보를 보장한다.<br>
결과적으로, PatcCore는 광범위한 메모리 뱅크를 통해 테스트시 사용 가능한 정상 정보를 최적으로 활용할수 있다.<br>
마지막으로, PatchCore는 추출된 패치 레벨 메모리 뱅크의 중복성을 줄이고, 스토리지 메모리와 추론 시간 단축을 위해 탐욕적 coreset 서브샘플링을 추가로 도입하였다.
따라서 PatcCore는 매우 실용적이고, 산업에 활용하기 매력적이다.



