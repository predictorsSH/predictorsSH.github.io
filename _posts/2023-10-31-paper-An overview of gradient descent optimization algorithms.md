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
모든 데이터를 사용해 비용함수에 대한 파라미터의 gradient를 계산한다.<br>

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
- 여기서 $ \gamma $는 1이하의 값을 가진다.(논문에서는 0.9)

momentum의 식을 보면, 기존 경사하강법 보다 $ \gamma v_{t−1}$ 만큼 추가적으로 더 이동 하는 것을 알 수 있다.<br>
$ \gamma v_{t−1} $ 는 이전 step 에서의 기울기이므로 이전에 이동하던 방향으로 조금 더 이동 하는 것이라 볼 수 있다.<br>
정리하면 현재 gradient와 이전의 Gradient 경향(방향)을 모두 감안하여 현재 업데이트 방향과 크기를 결정하는 것이다.


결과적으로 위 두번째 그림과 같이 그래디언트가 동일한 방향을 가리키는 차원에 대한 모멘텀항은 증가하고, 그래디언트 방향이 변경되는 차원에
대한 업데이트는 감소한다. 

#### 문제점
- momentum 영향으로 최적해를 지나칠 수 있다. 

### Nesterov accelerated gradient

NAG는 gradient 계산 순서를 바꿔서 Momentum의 문제를 해결하려고함.

- $ v_t = \gamma v_{t−1}$ + $η∇_{θ}J(θ- \gamma v\_{t-1})$
- $ θ = θ − v_{t} $

위 식 NAG 식에서 Momentum과 차이는  $η∇_{θ}J(θ- \gamma v\_{t-1})$ 이 부분이다.<br>
Gradient를 구할 때 momentum term 크기 만큼 이동한 후 구하겠다는 것이다.<br>

![momentum_NAG](/assets/images/contents/paper/gradient descent optimizer/Momentum_NAG.PNG)


### Adagrad

이전에는 모든 파라미터에 대해 동일한 학습률을 사용해 학습했다.<br>
그런데 어떤 파라미터는 최적값에 거의 도달했고, 다른 값은 멀리 떨어져있을 수 있다.<br>

Adagrad는 지금까지 많이 업데이트된 파라미터에 대해서는 작은 업데이트를 수행하고, 적게 업데이트 된 파라미터에 
대해서는 큰 업데이트를 수행한다.<br>


Adagrad는 희소한 데이터를 처리하는데 적합하다.<br>

- $ g_{t,i} = ∇_{θ}J(θ\_{t,i}) $ 

Adagrad는 위 식 처럼 gradient를 구할 때 각 parameter $ θ_{i} $ 마다 gradient를 계산한다.<br> 

- $θ_{t+1,i} = θ_{t,i} - { η \over \sqrt{G_{t,ii} + \epsilon} } · g_{t,i} $

위 식에서 Adagrad가 파라미터를 어떻게 업데이트 하는지 볼 수 있다.<br>
학습률의 분모에 $ \sqrt{G_{t,ii} + \epsilon} $이 들어가는 것 빼고는 SGD와 같은 방법이다.

여기서 $ G_{t,ii} $ 는 t 시점까지 $ θ_{i} $ 에 적용된 gradient의 제곱합이다.<br>
아래 식을 보면 이해하기 쉽다.

- $ G_{t} = G_{t-1} + (∇_{θ}J(θ))^2 $ 

$ G_{t,ii} $ 가 크다는 말은 이때까지 파라미터의 gradient가 크게 변화해왔단 의미이다.<br>
따라서 학습률에 $ G_{t,ii} $ 의 역수를 곱해주면, 그동안 많이 변해온 파라미터의 학습률은 줄어들게 된다.<br>
반대의 경우엔 학습률이 늘어난다.

Adagrad의 식을 행렬연산으로 표현하면 아래와 같다.<br>

- $θ_{t+1} = θ_{t} - { η \over \sqrt{G_{t} + \epsilon} }  \bigodot  g_{t} $

$ G_{t} $는 대각행렬로, 각 대각(i,i)원소에 $ θ_{i} $의 gradient 합이 누적된다.


### RMSprop/Adadelta

Adagrad는 학습 할수록 학습률이 작아지는 현상이 발생한다. <br>

이 문제를 해결하기 위해, RMSprop는 지수 가중 평균을 사용해 gradient를 누적한다.<br>
지수 가중 평균을 사용하면, 과거의 gradient의 영향이 지수적으로 감소하게 된다.
그리고 과거의 모든 graduents를 계산하는 대신에, 누적되는 과거 그레디언트의 window를 일정한 크기로 제한하게 된다.<br>
w개 이전의 제곱 그레디언트를 저장하는 대신, 지수가중평균으로 구현한다.<br>

- $ E\[g^2]_{t} = \gamma E\[g^2]\_{t-1} + (1- \gamma)g^2\_{t} $
- $ \gamma =0.9 $

위와 같이 gradient 제곱의 가중 평균을 구하면, 먼 과거의 gradient에는 작은 가중치가 곱해진다.

- $θ_{t+1} = θ_{t} - { η \over \sqrt{E\[g^2]\_{t} + \epsilon} } · g_{t} $

### Adadelta

Adadelta는 RMSprop와 같이 지수가중평균을 사용해 gradient를 누적한다.<br>
RMSprop와의 차이점은 **가중치 업데이터 단위를 맞추는 작업이 더해진다는 것이다**

??? 뭔말임


### Adam

momentum과 adaptive 모두 사용하자!

- $ m_{t} = β_{1}m_{t−1} + (1 − β_{1})g_{t} $

Adam의 momentum 식이다. 앞서 Momentum 공식과 유사하지만, 지수 가중 평균이 적용 된 것을 볼 수 있다.

- $ v_{t} = β_{2}v_{t−1} + (1 − β_{2})g_{t}^2 $

위 식은 Adative learning rate를 적용하는 것으로 기존 RMSprop 나 adadelta와 동일하다.
  
Adam의 파라미터 업데이트를 보면 아래와 같다.
- $θ_{t+1} = θ_{t} - { η \over \sqrt{ \hat{v}\_{t} + \epsilon} } · \hat{m}_{t} $
- $ \hat{m}_{t} = {m\_{t} \over 1-β\_{1}^t}  $
- $ \hat{v}_{t} = {v\_{t} \over 1-β\_{2}^t}  $

$ \hat{m}_{t} \hat{v}\_{t} $ 이 사용된 이유는, 편향을 보정하기 위해서이다.<br>

$ v_{t} $ 로 예를 들면, $ v_{0} $ 은 0으로 초기화 되고, $ v_{1} $ 은 $ 0.999*v_{0} + 0.001\*v_{1} $ 이 된다.<br>
이렇게 되면 $ v_{t} $ 의 초기 값들이 매우 작아지게 되는데 이를 보정하기 위해 $ v\_{t} $ 를 $ 1-β\_{2}^t  $ 로 나눠준다. <br>
t가 커질수록 $ 1-β\_{2}^t $ 값은 1에 가까워지기 때문에 t가 충분히 커지면 편향 보정의 효과는 거의 사라진다.

### AdamX

Adam 에서 $ v_{t} $ 를 계산할때 gradient의 L2 norm을 이용한다.<br>

- $ v_{t} = β_{2}v_{t−1} + (1 − β_{2})g_{t}^2 $

이떄 L2 norm 대신 p norm으로 일반화할 수 있다.
- $ v_{t} = β_{2}^pv_{t−1} + (1 − β^p_{2}) \left\| g_{t} \right\| ^p $

큰 p 값에 대한 놈(norm)은 일반적으로 수치적으로 불안정해지는 경향이 있어서 실무에서는 1과 2 놈이 가장 일반적이다.
그러나 ∞ 놈도 일반적으로 안정적인 동작을 나타낸다.

- $ u_{t} = β_{2}^\infty v_{t−1} + (1 − β^\infty_{2}) \left\| g_{t} \right\| ^\infty $ <br>
         $= max(β_{2} · v_{t-1}, \left\| g_{t} \right\|) $

- $θ_{t+1} = θ_{t} - { η \over u_{t} } · \hat{m}_{t} $

### Nadam

momentum 대신 NAG와 adaptive를 결합하면 성능이 더 좋지 않을까?

먼저 NAG의 파라미터 업데이트를 보자<br>

- $ g_{t} = ∇θ_{t}J(θ_{t}- \gamma m_{t-1} )  $
- $ m_{t} = \gamma m_{t-1} + ηg_{t}  $
- $ θ_{t+1} = θ_{t} - m_{t} $

Nadma에서는 NAG를 아래와 같이 수정한다.<br>

$$ g_{t} = ∇θ_{t}J(θ_{t})  $$
$$ m_{t} = \gamma m_{t-1} + ηg_{t}  $$
$$ θ_{t+1} = θ_{t} - (\gamma m_{t} + ηg_{t} ) $$

위에 두 식은 NAG방식이 아닌 Momentum 방식을 사용했고,<br> 
마지막 파라미터를 업데이트 할때 $ \gamma m_{t-1} $ 대신 $ \gamma m_{t} $를 적용한다. <br>

Adam에 Nesterov momentum을 적용시키기 위해, 이전 momentum vector를 현재 momentum vector로 바꿔준 것이다.

최종적으로 Nadam 식은 다음과 같이 정리할 수 있다.

- $ θ_{t+1} =  θ_{t} - {η \over { \sqrt{\hat{v}\_{t} } +\epsilon }} (β_{1} \hat{m}_{t} +  { (1- β\_{1})g\_{t} \over (1-β^t\_{1})})   $
  

