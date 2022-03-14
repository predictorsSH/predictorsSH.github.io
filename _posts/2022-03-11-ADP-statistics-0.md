---
layout: post
title: ADP-Statistics-기초통계량-1
subtitle: 실전 기초통계량 
author: san9hyun
categories: ADP
banner : /assets/images/banners/data.jpg
tags: ADP statistics 기초통계량 평균 중앙 왜도 첨도 EDA
---

## 🔑 기초 통계량

데이터 분석의 첫번째 지점은 기초통계량을 살펴보는 것이다.
먼저, 어떤 통계량들이 있는지 확인하고, 실제 데이터에서 기초 통계량을 도출하고 시각화 해보자.

### 기초통계량
기초 통계량이란, 말그대로 주어진 데이터의 가장 기초적인 특징을 알려주는 값들이다.

- 데이터 개수(표본 개수)
- 평균(표본 평균)
- 분산
- 표준편차
- 최솟값, 최대값
- 중간값
- 분위수
- 최빈값

등이 기초통계량이라 할 수 있다.<br>
각 기초통계량에 대한 설명은 생략하도록 하고, 실제 데이터셋으로 기초통계량들의 값을 구해보자!

## 🔧 준비물(데이터 셋, colab)
먼저 데이터 셋을 준비하여야한다.<br>
나는 kaggel에 [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/overview) 대회에서 제공하는 데이터셋을 활용할 것이다.<br>

해당 대회 대해서 간단하게 설명하자면, 시간, 날씨, 기온 등의 정보를 가지고 자전거 대여 건수를 예측하는 대회이다.

이 데이터 셋을 준비한 이유는,<br>
1. 시계열 데이터이기 때문이다. 시계열 데이터를 다루는 문제가 ADP에 자주 출제 되기 때문에 연습해 둘 필요가 있다고 생각하였다.
2. 해당 데이터를 분석한 자료가 많다. 캐글 코드공유에만 해도 꽤 많은 정보가 있다.  
3. 마지막은 데이터가 비교적 쉽고 간단하기 때문이다. 데이터 파악하는데 애쓰다보면, 정작 필요한 공부가 힘들어진다!

당분간은 이 데이터를 가지고 놀면서, ADP 준비를 해볼 것이다.
그리고 분석 도구로는 python(colab)을 활용할 것이다.

colab에서 작성한 코드를 포스트할때는, 코드블럭 'python' 으로 작성하였다.(당연)
<br>
그리고 그 코드의 output은 코드블럭 'text'로 작성하였다.

## 🐌 데이터 분석

본격적으로 데이터 분석을 해보자!<br>
먼저, kaggle에서 받은 데이터 셋을 구글드라이브 data 폴더에 저장하고 <br>
colab에서 그 데이터를 불러온다.

```python
#드라이브 연동
from google.colab import drive
drive.mount('/content/drive/')

#작업폴더로 이동
cd "/content/drive/My Drive/Colab Notebooks/ADP/"

#필요라이브러리 임포트
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#데이터 load
train=pd.read_csv("data/train.csv")
test=pd.read_csv("data/train.csv")
```

### 필드 설명

먼저, 데이터를 이루고 있는 필드(속성)을 간단하게 소개하면 다음과 같다.

datetime
- hourly date + timestamp 

season
- 1 = spring,
- 2 = summer,
- 3 = fall,
- 4 = winter

holiday
- whether the day is considered a holiday

workingday
- whether the day is neither a weekend nor holiday

weather
- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain +    Scattered clouds
- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

temp
- temperature in Celsius

atemp
- "feels like" temperature in Celsius

humidity
- relative humidity

windspeed
- wind speed

casual
- number of non-registered user rentals initiated

registered
- number of registered user rentals initiated

count
- number of total rentals, target 변수

### 기초통계량 살펴보기

```python
train.info()
```
dataframe 형태로 데이터를 로드하면, 가장 먼저 입력하는 명령어는 dataframe.info()일 것이다.<br>
아래와 같이 컬럼명, 데이터 개수, 결측치, 데이터 타입, 메모리 사용 등에 대한 간단한 정보를 얻을 수 있다.
```text
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    10886 non-null  object 
 1   season      10886 non-null  int64  
 2   holiday     10886 non-null  int64  
 3   workingday  10886 non-null  int64  
 4   weather     10886 non-null  int64  
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64  
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64  
 10  registered  10886 non-null  int64  
 11  count       10886 non-null  int64  
dtypes: float64(3), int64(8), object(1)
memory usage: 1020.7+ KB
```

```python
train.describe()
```

pandas의 dataframe에서 가장 손쉽게 기초 통계량을 확인할 수 있는 방법은, <br> dataframe.describe() 함수를 사용하는 것이다.<br>
아래 표와 같은 형태로, 각 컬럼별 mean,std, 분위수 등 기초통계량을 얻을 수 있다.

  |index|season|holiday|workingday|weather|temp|atemp|humidity|windspeed|casual|registered|count|
  |---|---|---|---|---|---|---|---|---|---|---|---|
  |count|10886\.0|10886\.0|10886\.0|10886\.0|10886\.0|10886\.0|10886\.0|10886\.0|10886\.0|10886\.0|10886\.0|
  |mean|2\.5066139996325556|0\.02856880396839978|0\.6808745177291935|1\.418427337865148|20\.230859819952173|23\.65508405291192|61\.88645967297446|12\.799395406945093|36\.02195480433584|155\.5521771082124|191\.57413191254824|
  |std|1\.1161743093442644|0\.16659885062471985|0\.4661591687997421|0\.6338385858190921|7\.791589843987506|8\.474600626484888|19\.245033277394786|8\.16453732683849|49\.9604765726498|151\.0390330819246|181\.14445383028496|
  |min|1\.0|0\.0|0\.0|1\.0|0\.82|0\.76|0\.0|0\.0|0\.0|0\.0|1\.0|
  |25%|2\.0|0\.0|0\.0|1\.0|13\.94|16\.665|47\.0|7\.0015|4\.0|36\.0|42\.0|
  |50%|3\.0|0\.0|1\.0|1\.0|20\.5|24\.24|62\.0|12\.998|17\.0|118\.0|145\.0|
  |75%|4\.0|0\.0|1\.0|2\.0|26\.24|31\.06|77\.0|16\.9979|49\.0|222\.0|284\.0|
  |max|4\.0|1\.0|1\.0|4\.0|41\.0|45\.455|100\.0|56\.9969|367\.0|886\.0|977\.0|


다만, numeric 데이터 타입의 필드별로만 기초통계량을 확인 할 수 있고, <br>각 필드의 특정 그룹에 대해서 기초 통계량을 확인하기 위해서는, 추가적인 데이터 조작이 필요하다.

해당 데이터의 경우,필드 별로 기초 통계량을 확인해서는 큰 정보를 얻지 못한다.<br>
위에서 필자가 얻은 정보는<br>

casual, registered, count의 경우 중앙값과 평균값 차이가 크다는 것이다. <br> 이러한 정보는 세가지 필드들의 분포가 고르지 않다는 것을 알려준다.<br>

데이터 분석을 위해 더 확인해봐야 할 것은, 서로 다른 필드들의 관계이다. 특히 target 변수와의 관계를 파악하는 것이 중요하다.<br> 


### 연도별로 target 변수의 평균, 중앙값, 최대값, 최소값, 표준편차 등 기초 통계량 알아보기
특정 년, 월 별로 count 값의 평균을 알아보고자 한다. <br>
먼저, 데이터를 년, 월로 구분하고 그룹화 할 수 있어야한다.
object type의 'datetime'필드를 datetime type으로 변경하고, 년, 월, 일, 시간을 분리해 필드로 만들자.

```python
# 데이터 타입 변경
train['datetime'] = pd.to_datetime(train['datetime'])

# 년도, 월, 일, 시간으로 필드 분리
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.month
train['hour'] = train['datetime'].dt.hour
```

년, 월,일, 시간 컬럼을 생성해주었다.
이제 년도별, 월별 count의 평균이 어떻게 다른지 확인해보자

먼저, dataframe['col'].unique() 로 특정 컬럼에 어떤 값들이 있는지 확인한 후,
조건절로 인덱싱하여 평균을 구할 수도 있다. 그러나 더 간단한 방법은 groupby를 활용하는 것이다.

```python
print('년도:', train['year'].unique())
print('2011년:', train[train['year']==2011]['count'].mean())
print('2012년:', train[train['year']==2012]['count'].mean())
```
```text
년도: [2011 2012]
2011년: 144.223349317595
2012년: 238.56094436310394
```

```python
#groupby를 활용
train[['year','count']].groupby(by='year').mean()
```

  
  |year|count|
  |---|---|
  |2011|144\.223349317595|
  |2012|238\.56094436310394|

2011년도와, 2012년도의 평균값 차이가 매우 큰것을 알 수 있다.
그러나 평균값만을 가지고 섣부르게 판단할 수 없다.
이상치가 존재하지는 않는지 확인 해볼 수도 있고, 중앙값을 확인해 보는 것도 좋다.

```python
train[['year','count']].groupby(by='year').median()
```


  |year|count|
  |---|---|
  |2011|111\.0|
  |2012|199\.0|


중앙값의 차이도 매우 큰것을 알 수 있다.
이는 실제로, 2011년도와 2012년도의 count 평균값 차이가 이상치때문이 아니라는 주장에 힘을 실어준다.

조금더 데이터를 살펴보기 위해,
2011년도와 2012년도의 데이터 개수, count 최대값,count 최소값, count 표준편차등을 확인해보자

```python
print('    <counts>')
print(train[['year','count']].groupby(by='year').count())
print('    <max>')
print(train[['year','count']].groupby(by='year').max())
print('    <min>')
print(train[['year','count']].groupby(by='year').min())
print('    <std>')
print(train[['year','count']].groupby(by='year').std())
```
```text
    <mean>
        casual  registered       count
year                                  
2011  28.73792  115.485430  144.223349
2012  43.25000  195.310944  238.560944
    <median>
      casual  registered  count
year                           
2011    13.0        91.0  111.0
2012    20.0       161.0  199.0
    <max>
      casual  registered  count
year                           
2011     272         567    638
2012     367         886    977
    <min>
      casual  registered  count
year                           
2011       0           0      1
2012       0           1      1
    <std>
         casual  registered       count
year                                   
2011  39.554419  108.847868  133.312123
2012  57.584101  174.709050  208.114003
```
위 기초통계량들을 확인해보면 casual, registerd 중 registerd의 증가폭이 더 커보인다.
2012년도에 자전거 렌탈 서비스가 더 알려지면서, 가입한 사람이 증가하였다고 생각해볼 수 있다.

### 월 별로 target변수 기초통계량 알아보기/시각화

year는 2개의 유니크한 값을 가지는 반면에,<br> 
month는 12개의 유니크한 값을 가지기 때문에,<br>
위에서 처럼 표로 결과를 확인하면 한눈에 이해하기 힘들다.<br>
이를 해결하기 위해 시각화를 진행해보자.

먼저, 평균값에 대해서 시각화를 해보자

```python
#월별 count 평균이 저장되어 있는 새로운 데이터 프레임 정의
month_mean=train[['month','count']].groupby(by='month').mean(

plt.figure()
month_mean.plot(kind='bar')
plt.title('month_mean')
plt.show()
```


![month_mean](/assets/images/contents/ADP_statistics/month_mean.PNG)


위 막대그래프를 보면 6,7,8,9,10월에 바이크 렌탈 서비스를 이용한 고객이 많아보인다.
그럼 여기서 궁금증이 생긴다. 2011, 2012년도에 상관없이 위와 같은 모양의 그래프가 그려질까?


```python
#11년도, 12년도 따로 데이터프레임 생성
month_mean_11=train[train['year']==2011][['month','count']].groupby(by='month').mean()
month_mean_12=train[train['year']==2012][['month','count']].groupby(by='month').mean()
```
```python
plt.figure()
month_mean_11.plot(kind='bar')
plt.title('month_mean_11')
plt.show()
```

![month_mean](/assets/images/contents/ADP_statistics/month_mean11.PNG)

```python
plt.figure()
month_mean_12.plot(kind='bar')
plt.title('month_mean_12')
plt.show()
```


![month_mean](/assets/images/contents/ADP_statistics/month_mean12.PNG)

그래프의 차이가 있지만, 6월 7월, 8월, 9월이 다른 달보다 count가 높은 경향이 있다!!!
그렇다면 월별 날씨의 영향으로 count의 차이가 나는 것은 아닐까?라고 추측해볼 수 있다.
그럼 이제month와 날씨(weather, temp 등)의 관계를 살펴보자!

잠시! 그전에! target변수의 기초통계량에서, mean과 median이 크게 차이가 났었다.
target 변수의 분포를 확인해보기 위해 히스토그램을 그려보고 왜도와 첨도를 구해보자!

### target변수 왜도 첨도 구하기

왜도(Skewness) : 분포의 비대칭도를 나타내는 통계량이다. 정규분포, T분포와 같이 대칭인 분포의 경우 왜도가 0이다. <br>

- 왼쪽으로 치우침(오른쪽 꼬리가 김) : 왜도 > 0
- 오른쪽으로 치우침(왼쪽 꼬리가 김) : 왜도 < 0

![skewness](/assets/images/contents/ADP_statistics/skewness_0.PNG "출처:https://dining-developer.tistory.com/17(skewness)")

첨도(Kurtosis) : 첨도는 분포의 꼬리부분의 길이와 중앙부분의 뾰족함에 대한 정보를 제공하는 통계량이다.

- 정규분포 : 첨도 0(Pearson 첨도 = 3)
- 위로 뾰족함 : 첨도 > 0
- 아래로 뾰족함 : 첨도 < 0 


![kutosis](/assets/images/contents/ADP_statistics/kutosis_0.PNG "출처:https://dining-developer.tistory.com/17(skewness)")

왜도 첨도의 자세한 설명은 [이 블로그](https://dining-developer.tistory.com/17)를 참고하자!

pandas로 왜도와 첨도를 구하는 방법은 간단하다.<br>
dataframe['col'].skew() dataframe['col'].kurt()를 각각 입력하면 된다.

```python
#왜도 구하기, 첨도 구하기
print('왜도:',train['count'].skew()) 
print('첨도:',train['count'].kurt())
```
```text
왜도: 1.2420662117180776
첨도: 1.3000929518398334
```
왜도와 첨도가 모두 1보다 큰 것을 알 수있다.<br>
아마 분포그래프를 그리면 오른쪽 꼬리가 길고 중앙이 뾰족한 그래프가 그려질것이다.<br>
seaborn library를 활용해 그래프를 그려보자!<br>
seaborn의 distplot() 은 히스토그램을 그려준다!

```python
sns.displot(train['count'])
```

![skewness](/assets/images/contents/ADP_statistics/skewness.PNG)

그래프가 정규성을 보이지 않는다! 이러한 경우 정규화 과정을 거쳐줘야 한다.
Skewed 되어있는 값을 그대로 학습시키면 꼬리 부분이 상대적으로 모델에 영향이 거의 없이 학습되기 때문이다.


## 🎬 다음 이야기
다음 포스트에서는, 다른 필드들을 활용해서 또 다른 기초통계량에 대해서 살펴볼 것이다.
