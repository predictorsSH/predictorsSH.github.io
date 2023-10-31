---
layout: post
title: An overview of gradient descent optimization algorithms(작성중)
subtitle: 경사하강법 최적화 알고리즘 
author: san9hyun
categories: paper
banner : /assets/images/banners/book.jpg
tags: DataScience gradient optimizer 
---

## 1.Introduction

### 논문의 목적
- 다양한 경사하강법 최적화 알고리즘들을 직관적으로 이해

### 논문 내용
- 경사하강법 최적화를 다루는 다양한 알고리즘 소개
- 각 알고리즘별 문제점/다른 알고리즘 등장 배경 소개

## 2.Gradient descent variants
목적함수의 경사(기울기)를 계산하기 위해 사용하는 데이터의 양에 따라 3가지 종류의 경사하강법이 있음

### Batch gradient descent 
모든 데이터를 사용해 비용함수에 대한 파라미터의 기울기를 계산한다.<br>

- θ = θ − η · $∇_{θ}$J(θ)

#### 문제점

- 메모리에 모든 데이터를 올려야함. 
- 새로운 데이터를 실시간으로 학습하는 것이 불가능.(온라인 학습 불가능)
- 한번의 업데이트가 매우 느리다.
- 데이터셋에 유사한 데이터가 있을경우 중복 계산 수행.(크리티컬한 문제로 보이지 않는다) 

### Stochastic gradient descent

각 샘플 $ x^{(i)} $, $ y^{(i)} $ 을 사용해 파라미터 업데이트 수행. <br>
매 에포크마다 훈련 데이터를 섞음. 

- θ = θ − η · $∇_{θ}$J(θ; $ x^{(i)} $ ; $ y^{(i)}$)

#### 장점

- Batch gradient descent의 중복 계산 문제 해결)
- 일반적으로 Batch gradient보다 빠름
- 온라인 학습 가능

#### 문제점/한계

- 목적함수 수렴이 오래 걸릴 수 있음(목적 함수의 변동이 심함, 아래 그림 참고)
- 벡터 연산의 장점을 활용 못함

![SGD fluctuation](/assets/images/contents/paper/gradient descent optimizer/SGD.PNG)

## 3.Challenges

위 경사하강법에서 해결되어야할 몇 가지 도전과제가 있다. 

- 적절한 학습률 선택이 어렵다.
  - learning rate schedule는 사전에 정의된 알고리즘으로, dataset 특성에 스스로 적응하는 알고리즘이 아니다.
  - learning rate schedule은 모든 파라미터에 같은 학습률을 적용 시킨다.
- local minima와  saddle point에 빠질 위험이 있다.

## 4.Gradient descent optimization algorithms

### Momentum

SGD는 샘플데이터 하나에 대해서 목적함수의 기울기를 계산하고 파라미터를 업데이트 하기 때문에, 최적해로 빠르게 수렴하지 않는다.<br>
그리고 local minima에 빠질 우려가 있다. (기울기가 0에 가까운 평탄한 지역에서 느리게 벗어남)

위 그림에서 우리는 수직축으로는 더 작은 학습율을 원하고, 수평 축에 대해서는 더 빠르게 학습하길 원한다.<br>

momentum은 최적해의 방향으로 SGD를 가속시켜준다.

![momentum](/assets/images/contents/paper/gradient descent optimizer/momentum.PNG)

- $ v_t = \gamma v_{t−1}$ + $η∇_{θ}$J(θ)
- $ θ = θ − v_{t} $

momentum의 식을 보면, 기존 경사하강법 보다 $ \gamma v_{t−1}$ 만큼 추가적으로 더 이동 하는 것을 알 수 있다.<br>
$ \gamma v_{t−1} $ 는 이전 step 에서의 기울기이므로 이전에 이동하던 방향으로 조금 더 이동 하는 것이라 볼 수 있다.  
여기서 $ \gamma $는 1이하의 값을 가진다.(논문에서는 0.9)

결과적으로 위 두번째 그림과 같이 그래디언트가 동일한 방향을 가리키는 차원에 대한 모멘텀항은 증가하고, 그래디언트 방향이 변경되는 차원에
대한 업데이트는 감소한다.

### Nesterov accelerated gradient

