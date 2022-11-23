---
layout: post
title: 수많은 데이터에 대한 경사 하강을 어떻게 빠르게 할수 있을까?
subtitle: 로지스틱 회귀 계산의 벡터화 
author: san9hyun
categories: AI
banner : /assets/images/banners/sea-gec.jpg
tags: DataScience neuralnet
---

* 앤드류 응 교수님 강의를 듣고 정리 하였습니다. 

신경망을 학습시키기 위한 순전파, 역전파 연산은 계산량이 많다.<br>
텐서플로우, 파이토치는 크기가 큰 데이터셋을 어떻게 빠르게 학습까?<br>
for문 대신 벡터 연산을 수행하기 때문이다.

## for문 vs 벡터연산

간단하게 for문을 사용했을때와, 벡터연산을 했을때 속도 비교를 해보자<br>
100만 크기의 벡터 두개에 대해서 벡터 곱을 수행할 떄, 속도 비교 결과는 아래와 같다.

![vectorize](/assets/images/contents/NN/vectorize/vectorize.png)

벡터연산은 3밀리초가 걸렸지만, for문을 활용했을때는 500밀리초가 걸렸다.<br>
for문이 100배가 넘는 시간이 걸리는 것을 확인할 수 있다.<br>

신경망 내부 연산을 수행할 때도, 벡터 연산으로 계산에 드는 시간을 크게 절약 할 수 있다.<br> 
신경망에서는 어떻게 for문 대신, 벡터 연산을 수행할 수 있는지 알아보자.<br>

간단한 로지스틱 회귀를 구현한 신경망이 있다고 생각해 봤을 때, 아래와 같이 연산 과정을 요약할 수 있다. 

## for 문을 활용 

- $ J = 0 $ Loss 
- $ dw= 0 $ (J에 대한 w의 도함수, w는 weight)
- $ db = 0 $ (J에 대한 b의 도함수, b 는 bias)
- y는 실제값, 라벨
- a는 모델의 예측값, 순전파 결과 값

위와 같이 변수를 정의하고, 각 변수를 0으로 초기화 했을 때,<br>
m 개의 샘플을 가지고, n_x 개의 피처를 가진 데이터로 로지스틱 회귀 모델을 학습 한다고 생각하자.<br> 

학습 과정을 표현하면 아래와 같다.<br>

```text
for i = 1 to m: #m개의 샘플
  
  z = w.T * x[i] + b #순전파
  a = sigmoid(z) #순전파
  J += -[y[i]*log(a)+(1-y[i])*log(1-a)] # 로지스틱 회귀 비용함수
  dz = a - y[i] # dz 는 J에대한 z의 도함수, 역전파
  
  for j = 1 to n_x:
    dw[j] += x_j*dz
  db += dz # 역전파

J = J/m, dw_i=dw_i/m, db = db/m # 샘플 개수로 평균 

```

두번의 for문이 사용 되는 것을 볼 수 있다.<br>
모든 샘플에 대한 연산을 하기 위해 첫번째 for문이 필요하고, 두번째 for문은 n_x개의 피처에 대한 weight을 업데이트 할때 필요하다.<br>

간단한 신경망을 학습시킬때도 우리는 몇 만개 이상의 데이터를 사용하고, 피처수도 1000개를 넘어 갈 때가 많다.<br>
그래서 for 문의 사용하게 되면, 계산에 많은 시간이 걸릴 것이다.

## 벡터화
벡터화를 통해, for 문 없이 m개의 샘플, n_x개의 피처에 대한 연산을 수행할 수 있다.


- $ dw= [0,0,...] $ (J에 대한 w의 도함수, w는 weight)
- $ db $ J에 대한 b의 도함수, b 는 bias
- $ z = [z_i, ..., z_m] $
- $ X = [X_i, ..., X_m] $
- $ X_i = [X1, X2, X3, ..., Xn_x] $ (n_x개의 피처가 있을떄) 
- $ Y = [y_i, ..., y_m] $ (살제 값, 라벨)


```text
Z = dot(W.T , X) + b 
A = sigmoid(Z) #순전파 
J = -1 * reduce_mean([Ylog(A) + (1-Y)*log(1-A)])
dZ = A - Y 
dw = reduce_mean(dot(X , dZ.T))
db = reduce_mean(dZ)

w := w-lr*dw  #lr = learning rate
b := b-lr*db
```
위와 같이 벡터 연산을 통해서 for문 없이도 모든 데이터 샘플에 대한 연산이 한번에 가능하다.
경사하강법을 여러번 할 경우, 즉 학습을 여러번 할 경우에는 어쩔 수 없이 for문을 사용해야한다.
