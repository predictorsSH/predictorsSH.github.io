---
layout: post
title: Anomaly Detection in Time-Series Data (작성중)
subtitle: 시계열 이상탐지 자료 조사
author: san9hyun
categories: paper
banner : /assets/images/banners/book.jpg
tags: DataScience TimeSeries paper
---

## Deep Learning for Anomaly Detection in Time-Series Data: Review, Analysis, and Guidelines

### [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9523565)
KUKJIN CHOI, JIHUN YI, CHANGHWA PARK, AND SUNGROH YOON 

*참고자료*<br>
[고려대 산업경영공학부 세미나](https://www.youtube.com/watch?v=PZp99rCHw1c)


### Properties of Time-Series Data

- Temporality : 특정한 시점에서 관측된 데이터의 연속된 집합
- Dimensionality : 하나(univariate) 혹은 복수개(multivariate)의 특성을 가짐 
- Nonstationarity : 시간이 흐르면 데이터의 특성이 변함
- Noise : 데이터의 수집, 저장, 전달, 처리 등의 과정에서 발생하는 어쩔 수 없는 데이터의 변화

### Anomaly type 

- Point Anomaly : 갑작스럽게 일반적인 상태를 벗어나는 하나의 관측값 또는 연속된 여러 관측값
- Contextual Anomaly : 일반적인 범주를 벗어나지 않으나 비정상적 패턴이 관찰되는 경우
- Collective Anomaly : 점진적으로 정상과 다른 패턴을 보이는 연속된 데이터의 집합 (**한번에 검출하기 힘들어 장기적인 관찰 필요**)
- Other anomaly types : 정상을 어떻게 정의하냐에 따라, 여러가지 Anomaly 유형이 생겨날 수 있음. 
  
### Classical Approach

- Time domain : 측정된 폭과 높이 임계치를 이용
- Frequency domain : Fourier analysis, 복잡한 시계열을 주기함수(sine, cosine)의 결합으로 표현
- Statistical Model : 데이터에 대한 통계량(mean, variance, median 등)을 활용하여 모델링
- Distance Based : 두개의 temporal 시퀀스에 대한 거리를 계산하여 둘 사이의 유사도를 측정, Normal과 새로운 관측값 거리를 계산해 Anomaly 판별
- Predictive Model : 과거부터 현재까지 데이터를 사용하여 미래 상태를 예측. 예측된 값과 실제로 관측된 값의 불일치 정도를 계산하여 Anomaly 판별.
  - Autoregressive Integrated Moving Average(ARIMA)
- Clustering Model : 라벨링 되지 않은 데이터를 클러스터링을 수행하고, 클러스터 중심에서의 거리를 기준으로 새로운 관측값에 대한 Anomaly 판별
  - OCSVM, GMM, DBSCAN 등
  
### Challenging Issues
실제 산업 환경에서 이상탐지 모델을 구축하고 사용하기 힘들다. 대표적인 이유는 아래 두가지 이다.<br>

- 라벨 부족 :  이상 데이터는 실제 산업환경에서 극히 드물다. 
- 데이터의 복잡성 : 차원의 저주로 인해 전통적인 접근 방식은 성능이 저하됨.

위와 같은 문제를 해결하기 위해서 다양한 Deep Anomaly 기법들이 있다.
기법들은 보통 아래과 같은 단계를 거친다.<br>

1. 변수간 상관관계를 고려한 처리 (PCA, AE 등)<br>
2. 시계열 데이터 representaion (rnn, cnn, ConvLSTM)<br>
3. Anomaly Detection (reconstructor error, prediction error 등)<br>

1번 또는 2번의 경우 생략될 수도 있다. 아래에서 위 단계들을 차례로 살펴본다.

### Inter-correlation between variables
대부분 다변량 시계열 데이터를 위한 딥러닝 모델은 매 타임 스텝마다 변수들 사이의 관계를 설정한다. 
이러한 시공간 정보는 시간적 맥락뿐 아니라 변수간 상관관계도 고려한다.

#### Dimensional Reduction 
큰 스케일의 시스템의 상태는 중요한 요인 몇개만으로 설명될 수도 있다. 
차원 축소를 통해서 중요한 요인만 추출함으로써 계산량을 줄일 수 있다.
일반적으로, linear algebra based 방법, neural network based 을 많이 사용한다. 
- linear algebra-based : PCA, singular value decomposition
- nn : AE, VAE

#### 2D MATRIX
직접적으로 변수간 유사성과 상대적 스케일을 직접 캡처하는 방법<br>
$ X = {x^1,x^2,...,x^T} $ <br>
$ x^t = {x_1^t,x_2^t,...x_n^t}$ <br>
일때 아래와 같은 대표적인 두가지 메트릭스 표현법이 있다. 

![anomaly_type](/assets/images/contents/anomaly survey/2DMatrix.PNG)

첫번쨰 식은 특정 시점에 전체 변수가 갑자기 커지거나 작아질때를 감지할 수 있음<br>
두번째 식은 concept drift 또는 change point를 감지할 수 있음<br>

#### Graph
그래프는 명시적인 위상 구조를 정의하고, 개별 변수 간 인과 관계를 학습할 수 있음. 최근에는 Attention 매커니즘을 적용해서 성능을 향상시킴.

#### others
- raw data에서 직접 Anomaly Detection 수행
- Multivariate Gaussian distribution을 활용해 변수 간의 관계를 파악

![inter_correlation](/assets/images/contents/anomaly survey/inter_correlation.PNG)


### Modeling Temporal Context

#### RNN  
미래 데이터를 예측하고, 시퀀스 패턴을 인식하는 가증 흔한 모델 중 하나, 기울기 소실 문제를 해결하기 위해 GRU, LSTM 등장

#### CNN 
주로 이미지 처리에 사용되지만, 시계열 모델링에도 사용. Segmented series에 대한 pattern을 파악.

#### Hybrid 
RNN, CNN 함께 사용하는 방법. 예를들어, 차원이 30이고 3가지 종류의 window를 사용한다고 가정하면 
데이터의 shape은 (30,30,3)임(이미지와 동일).
여기서 5타입 스텝만큼의 데이터를 쌓아서 처리한다면 데이터는 (5,30,30,3)가 됨(비디오와 동일).
이러할때 사용할 수 있는 Hybrid 모델은 ConvLSTM이 있음.

#### Attention
Output에 기여하는 정도에 따라 피처에 가중치를 주는 기법. 

#### Hierarchical temporal memory(HTM)
별도 자료 조사 필요

### Anomaly criteria
위에서 다룬 모델들은 손실함수를 최소화하는 학습 방식으로 데이터의 표현을 학습한다. 
학습된 모델은 Anomaly Score를 출력하는데, 해당 값이 크면 이상 데이터일 확률이 높다.<br>

Anomaly Score는 세가지 유형으로 분류된다.

- Reconstruction Error : 정상데이터 재구축 모델을 학습하고, 새로운 데이터의 재구축 에러가 높으면 이상으로 탐지
- Prediction Error : Prediction model에서 예측값과 실제값 차이가 크면 이상으로 탐지
- Dissimilartiy  : 새로운 데이터가 학습된 군집 혹은 분포와 멀리 떨어져있으면 이상으로 탐지


![anomaly_score](/assets/images/contents/anomaly survey/anomaly_score.PNG)

### Guidlines for practitioners

#### Realtime vs Early Warning

- Realtime : 온라인 비즈니스나 금융처럼 실시간 이상탐지가 요구되는 경우. 사고에 대한 빠른 응답이 중요한 경우
  Reconstructor error를 사용하는 GRU, CNN based model 추천
  
- Early Warning : Anomaly 발생시 피해가 매우 큰 산업.
  Autoregressive 알고리즘을 사용하는 LSTM, HTM based model 추천 (미래 사고를 예측할 수 있는 모델)

#### Sliding Window vs Incremental Update
컨텍스트를 추론하기 위해서 모든 과거 데이터를 처리하거나, 최신 목에 대한 출력을 점진적으로 업데이트하는 방식이 있음

- Sliding Window : 모든 과거 data를 처리
  - 정해진 window size를 통해 data처리
  - 적절한 window size 결정하는 것이 중요
  - TCN, CNN based model 추천

- Incremental Update : 점진적으로 새로운 data에 대해 update
  - Marginal computation을 통해 새로운 data에 대해 예측을 수행
  - Data가 하나씩 들어오는 스트리밍 환경에 유리
  - GRU, LSTM based model 추천

#### Batch Learning vs Online Update
시계열데이터에서 공통적이 문제는 데이터의 비정상성이다. 데이터 분포의 변화를 따르면서 모델을 업데이트하기 위해 두가지 유형의 접근 방식을 제안한다.

- Batch Learning : 일반적인 학습방법. 데이터가 정상적이라고 가정하고 학습. 
- Online Update : 일반적으로 잘 사용되지 않음. 새로 추가된 데이터에 대해서 지속적으로 model의 학습을 이어나감.

#### Denoising
Noise는 Anomaly와 구분되기 힘들기 때문에 제거를 고려해야함. <br>
아래와 같은 방법들이 있음.

- Smoothing : 지수가중이동평균(현재의 관측치에 가중치를 주는 방법)
- Transformation : Wavelet transform or Fourier transform 를 통한 주파수 특성 추출. 추출 안된 것이 노이즈
- Estimation : Kalman Filter(노이즈가 있는 데이터를 상태공간 모델로 표현하고 확률론적 추정을 적용하여 노이즈 제거)
- deep_learning : Denoising AutoEncoder(무작위 노이즈를 추가하여 원래 입력을 복원하도록 훈련)

