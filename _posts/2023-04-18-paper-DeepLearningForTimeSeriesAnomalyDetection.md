---
layout: post
title: 시계열 데이터에서의 Aomaly 란
subtitle: Time Series Anomaly
author: san9hyun
categories: sundries
banner : /assets/images/banners/squirrel.jpg
tags: DataScience paper Anomaly 
---

## Time Series Anomaly

아래 논문에서 시계열 Anomaly 종류에 대해서 설명하는 부분만 정리

### [Paper](https://arxiv.org/pdf/2211.05244.pdf)

### Univariate Time Series(UTS)
시계열(time series)은 순차적인 시간을 인덱스로 가지는 데이터의 집합이다.
시계열은 단일 변수(univariate)와 다중 변수(multivariate)로 나누어진다.

UTS는 시간에 따라 하나의 변수가 변하는 시리즈 데이터이다. 
t 시간동안의 데이터 X는 다음과 같이 표현된다.<br>
$$ X = (𝑥_1, 𝑥_2, . . . , 𝑥_𝑡) $$ 

### Multivariate Time Series(MTS, 다변량 시계열)
MTS는 시간에 따라 여러개의 변수가 변하는 시리즈 데이터이다.
다변량 시계열에서 각 변수는 과거 값과 다른 변수에 영향을 받는다.
d개의 변수(차원)를 가진 데이터를 한 시점(t)에서 표현하면 다음과 같다.<br>
$$ X_t = (𝑥_t^1, 𝑥_t^2, . . . , 𝑥_𝑡^d) $$ 

### Time series decomposition 

*참고 여기 내용 내가 아는 사실이랑 다름.*  <br>
*Cyclical fluctuation 은 경기 변동과 같이 일정한 주기가 없는 장기적인 변동을 말하는 것으로 아는데 여기 설명은 다름.* 

시계열 X를 4가지 컴포넌트로 분해할 수 있다.

- Secular trend : 데이터가 장기적으로 상승 또는 하강하는 움직임이 있는 경우. 움직임이 꼭 선형일 필요는 없다. 
  예를 들어, 특정 지역의 인구 변화가 여러해 동안 역학적 요인에 따라 비선형적으로 중가 또는 감소하는 경우.
  
- Seasonal variations : 월, 주와 같이 특정 기간을 주기로 나타나는 패턴. 항상 고정된 빈도로 발생.
  예를 들어, 가스/전기 소비는 계절에 따라 패턴이 달라진다. 

- Cyclical fluctuations: 일정한 주기 없이 증가하거나 감소하는 경향. 예를 들어, 하루동안 온도 변화(**온도 변화는 Seasonal 아닌가?**)

- Irregular variations : 무작위적이고 불규칙한 사건. 다른 모든 구성요소가 분해된 후 나머지 부분. 

### Anomaly in Time Series/types of Anomalies

Anomaly는 일반적인 분포에서 크게 벗어난 하나의 관측 값 또는 시퀀스를 말한다. <br>
UTS와 MTS 이상치는 시간적(temporal), intermetirc(번역 모르겠음) 또는 시간적-intermetric 이상치로 구분할 수 있다.
단일 시계열에서 시간적 이상치는 local과 global 이상치로 구분된다. 시간적 이상치는 다변량 시계열에도 역시 나타 날 수 있다.

![anomaly_type](/assets/images/contents/anomaly survey/anomaly_types_univariate.PNG)

- Global : 다른 값들에 비해 극단적인 값을 가지는 시리즈 관측값들, 예를 들어 급격하게 감소 또는 증가하는 지점
- Contextual : 인점한 주변 데이터들과 편차가 큰 경우
- Seasonal : 계절성에서 벗어나는 경우. 
- Trend : 해당 데이터의 평균을 영구적으로 이동시키는 경우, 주기와 계절성은 유지하면서도 추세의 기울기(변화)가 급격하게 변하는 경우
- Shapelet: 전체 시퀀스와 다른 형태 또는 주기를 가지는 서브 시계열 
