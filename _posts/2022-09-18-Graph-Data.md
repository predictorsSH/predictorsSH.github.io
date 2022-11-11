---
layout: post
title: 회귀와 상관분석의 차이
subtitle: 면접질문
author: san9hyun
categories: interview 
banner : /assets/images/banners/post-bg.jpeg
tags: DataScience interview 
---

## 서론

Linear regression과 Correlation의 차이는 무엇인가요? 라는 인터뷰 질문을 받았다.<br>
평소 Linear regression과 Correlation이 전혀 다른 것이라고 생각하고 있었기 때문에, 질문에 다소 당황했다.<br>

당황한 탓인지, Linear regression과 Correlation의 차이에 대한 대답을 하지않고,<br>
회귀 계수와 상관 계수를 비교하는 엉뚱한 대답을 해버렸다.<br>

늘 그렇듯 인터뷰가 끝나고 다시 생각해보면,<br>
"회귀는 변수들 사이의 함수 관계를(회귀식?) 가정하고, 회귀 계수를 추정함으로써 변수들 사이의 관계를 설명하는 분석기법이고 상관분석은 두 변수 사이의 선형관계를 검정하는 분석이다." 라는 그럴듯한 답이 생각난다.

이 답이 그럴듯하기만 한 정답인지, 진짜 답에 가까운지 한번 찾아보았다.

## 회귀와 상관분석의 차이

우선 회귀와 상관분석은 변수들 사이의 의존성,관계를 분석하는 기법이라는 점에서만 어느정도의 공통점이 있고, 아래와 같이 여러 면에서 다른 분석 기법이었다.

|-|회귀|상관분석|
|---|---|---|
|목적|종속변수가 독립변수에 의존성을 가지는지 확인|종속, 독립변수 구분이 없음|
|인과관계|부분적 가능(어떻게?)|설명 불가능|
|예측성|독립변수에 의해 종속변수를 예측|없음|
|측정대상|회귀계수, 결정계수|상관계수|


아마, 면접관이 바라던 대답은 아래와 같은 답이었을까?

- 회귀분석 : 독립변수들이 종속변수에 영향력이 있는지, 그 크기는 어느정도인지 알아보는 분석기법, 영향력이 큰 독립변수들로 만든 회귀식으로 종속 변수를 예측하기도 함
- 상관분석 : 변수들이 얼마나 밀접한 관련성을 가지고 변화하는지 분석하는 기법 

## 회귀계수와 상관계수의 관계 

회귀분석은 두 변수 사이의 관계 정도를 회귀계수의 크기로 측정하고, 상관분석은 상관계수로 측정한다.<br>
그렇다면, 회귀계수와 상관계수에 어떤 관련성이 있을까?<br> 

만약 두 변수의 표준편차가 같다면,<br>
단순 회귀 분석에서 두 변수 사이의 회귀계수는, 단순 상관분석에서의 상관계수와 같다. <br>

아래 회귀계수의 분자와 분모를 n으로 나누면,

$$ \beta_1 = {\Sigma(x_i - \bar{x})(y_i - \bar{y}) \over \Sigma(x_i - \bar{x})^2}  $$

다음과 같이 분자는 $X,Y$ 의 공분산 분모는 $X$의 분산이 된다.<br>

$$ \beta_1 = {\Sigma(x_i - \bar{x})(y_i - \bar{y})*{1 \over n} \over \Sigma(x_i - \bar{x})^2 *{1 \over n} }  $$

$$ \beta_1 = {Cov(X,Y) \over \sigma(X)^2}  $$

상관계수를 위와 비슷하게 나타내 보면,<br>

$$ Coef(X,Y) = {Cov(X,Y) \over \sigma(X)\sigma(Y)} =  {Cov(X,Y) \over \sigma(X)^2} * {\sigma(X) \over \sigma(Y)}  $$

$$ Coef(X,Y) =  \beta_1 * {\sigma(X) \over \sigma(Y)}  $$

즉 상관계수는 회귀계수에, 두 변수의 표준편차의 비율을 곱한 것과 같으므로<br>
두변수의 표준편차가 같으면, 회귀계수와 상관계수는 같다.