---
layout: post
title: 시계열 데이터 시각화 with seaborn 
subtitle: Store Sales EDA 1
author: san9hyun
categories: EDA
banner : /assets/images/banners/squirrel.jpg
excerpt_image: /assets/images/contents/EDA/EDA-1_files/EDA-1_47_0.png
tags: EDA tutorial 시계열 seaborn 시각화
---

캐글 [Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) 대회 데이터를 분석하였습니다. 
해당 대회는 대규모 식료품 소매업체인 Corporación Favorita의 매장 매출을 예측하는 것을 목표로 합니다.
학습 데이터로 각 일자별 제품에대한 판매량을 제공합니다. 그리고 각 판매 매장에 대한 정보, 기름가격 변동 데이터, 휴일 이벤트 데이터도 별도로 제공됩니다.

분석할 거리가 많습니다!
다양한 시각화 기법을 활용해서, 해당 데이터를 분석해보겠습니다!

## 데이터 로드 / 필요한 라이브러리 임포트


```python
# 압축해제
!unzip -qq "/content/drive/MyDrive/Colab Notebooks/project/competition/캐글-Store Sales/store-sales-time-series-forecasting.zip"
```


```python
# 필요한 라이브러리
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
```


```python
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3000888 entries, 0 to 3000887
    Data columns (total 6 columns):
     #   Column       Dtype  
    ---  ------       -----  
     0   id           int64  
     1   date         object 
     2   store_nbr    int64  
     3   family       object 
     4   sales        float64
     5   onpromotion  int64  
    dtypes: float64(1), int64(3), object(2)
    memory usage: 137.4+ MB


> ⛳ 컬럼 정보
- date: 날짜
- store_nbr: 상점
- family: 제품군
- sales: 판매량
- onpromotion: 특정 날짜에 홍보중인 제품군 품목 수

## EDA (train.csv)

### 기초 통계량 (평균, 표준편차, 사분위수, 최빈값 등)

> 🧑 : 먼저 데이터의 기초 통계량을 확인하자.
> - categorical 변수는 unique(카테고리 개수), 최빈값, 최빈값의 빈도를 확인할 수 있다.
- numerical 변수는 평균, 사분위수, 최대값, 최소값, 표준편차를 확인 할 수 있다.


```python
# store_nbr은 categorical 변수로, 데이터 타입을 object로 변경
train_df['store_nbr'] = train_df['store_nbr'].astype('object')

train_df.describe(include='all')
```





  <div id="df-f1bc1433-4e43-4887-b1be-8b21a4b3ec17" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>store_nbr</th>
      <th>family</th>
      <th>sales</th>
      <th>onpromotion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.000888e+06</td>
      <td>3000888</td>
      <td>3000888.0</td>
      <td>3000888</td>
      <td>3.000888e+06</td>
      <td>3.000888e+06</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>1684</td>
      <td>54.0</td>
      <td>33</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>2013-01-01</td>
      <td>1.0</td>
      <td>AUTOMOTIVE</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>1782</td>
      <td>55572.0</td>
      <td>90936</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.500444e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.577757e+02</td>
      <td>2.602770e+00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.662819e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.101998e+03</td>
      <td>1.221888e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.502218e+05</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.500444e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.100000e+01</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.250665e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.958473e+02</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.000887e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.247170e+05</td>
      <td>7.410000e+02</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-f1bc1433-4e43-4887-b1be-8b21a4b3ec17')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-f1bc1433-4e43-4887-b1be-8b21a4b3ec17 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-f1bc1433-4e43-4887-b1be-8b21a4b3ec17');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-495eeebf-5169-486b-ba3f-3479f5832f02">
  <button class="colab-df-quickchart" onclick="quickchart('df-495eeebf-5169-486b-ba3f-3479f5832f02')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
width="24px">
<g>
<path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
</g>
</svg>
</button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-495eeebf-5169-486b-ba3f-3479f5832f02 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





### 학습 데이터 수집 기간 / 판매량 (Line Plot)

> 🧑 : 학습 데이터 수집 기간 동안 판매량이 어떻게 변화하는지 살펴보자.




> ⚡ 참고할 시각화 규칙
1. **테두리를 없애자!**<br>
   불필요한 테두리는 우리의 시각을 분산시킬 수 있다고 한다. <br>
   필요한 정보에 집중할 수 있도록 테두리를 지우자.<br>
2. **색은 필요할때만 쓰자!**<br>
   색은 강조하고 싶은 정보가 있을때만 사용하자!
   3.그래프와 제목 사이에 적절한 간격을 두자!








```python
# 데이트 컬럼 타입 변경
train_df['date'] = pd.to_datetime(train_df['date'],format = "%Y-%m-%d")
```


```python
# 주별로 데이터 집계. 데이터 개수가 너무 많기 때문에, 주, 월 년도 별로 미리 그룹화 하는 것이 좋음.
weekly_group_df = train_df.groupby([pd.Grouper(key='date', freq='W')]).agg(sales = ('sales', 'mean')).reset_index()
```


```python
# 시각화

fig, ax = plt.subplots(figsize=(12, 3))
# line plot
sns.lineplot(x='date', y='sales', data=weekly_group_df)
fig.suptitle('Sales Over Time', fontweight='bold')
fig.text(s=f"Start Date:{train_df['date'].min().strftime('%Y-%m-%d')}",
         x=0.20, y= 0.85, ha='center',fontsize=8)
fig.text(s=f" End Date:{train_df['date'].max().strftime('%Y-%m-%d')}",
         x=0.20, y= 0.80, ha='center',fontsize=8)

plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_14_0.png)



>☝ 해석
- 2013년부터 2017년 8월 15일까지의 데이터이다.
- 전체 sales가 조금씩 상승하는 경향이 있다. 데이터가 비정상적(Nonstational)이다. <br>

> ❎ 문제점 <br>
- 불필요한 색 <br>
  라인이 파란색이다. 그런데 위 그래프에서 색은 어떤 추가 정보도 가져다주지 못하므로 불필요하다.
- 불필요한 테두리<br>
  우측과 상단의 테두리는 우리의 시각이 그래프의 라인에 집중하는 것을 방해한다.<br>

>🧑: 위 그래프를 아래와 같이 다시 그려보자



```python
fig, ax = plt.subplots(figsize=(12, 3))
# line plot
sns.lineplot(x='date', y='sales', data=weekly_group_df, color='#808080')
fig.suptitle('Sales Over Time', fontweight='bold')
sns.despine(right=True, top=True)
fig.text(s=f"Start Date:{train_df['date'].min().strftime('%Y-%m-%d')}",
         x=0.20, y= 0.85, ha='center', color='#333333',fontsize=8)
fig.text(s=f" End Date:{train_df['date'].max().strftime('%Y-%m-%d')}",
         x=0.20, y= 0.80, ha='center', color='#333333',fontsize=8)

plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_16_0.png)



> ✅ 개선 <br>
- 작은 차이지만,
  그래프의 라인과 좌측 상단의 데이터 수집기간 정보가 더 눈에 들어온다.<br>

> 🧑: 앞으로는 모든 그래프에서 상단, 우측 테두리를 제거하겠다.


```python
# top, right 축 제거
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
```

### 판매량 경향 (Box Plot)

> 🧑: 연도별 Box Plot을 활용해서 데이터 분포의 변화를 확인할 수 도 있다.
- line plot은 시간에 따른 추이를 시각화 할때 효과적이다.
- box plot은 데이터의 분포가 시간에 따라 어떻게 변화하는지 확인할 수 있다.

> ⚡ 참고할 시각화 규칙
1. **이상치가 있을떄 시각화**<br>
   이상치의 영향을 제거하고 시각화하자.
2. **색은 적게 사용하자**<br>
   눈에 띄는 다양한 색상이 있으면 사용자가 데이터에서 의미를 추출하는 것이 더 어려워질 수 있다.


```python
train_df['year'] = train_df['date'].dt.year

sns.boxplot(y='sales', x='year', data=train_df)
fig.suptitle('Sales Over Time', fontweight='bold')
plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_21_0.png)



> ❎ 문제점 <br>
- 많은 이상치가 Sales의 평균값을 훨씬 웃돌기 때문에 Box Plot의 중요한 정보들이 보이지 않음


```python
# 이상치를 제거하기 위해 주별로 평균 내준 데이터를 사용
weekly_group_df['year'] = weekly_group_df['date'].dt.year

fig, axes = plt.subplots(1,2, figsize=(14,6))

sns.boxplot(y='sales', x='year', data=weekly_group_df, ax=axes[0])
sns.boxplot(y='sales', x='year', data=weekly_group_df, color='#B0E0E6', ax=axes[1])
fig.suptitle('Year-wise Plot', fontweight='bold')
axes[0].set_title("different colors")
axes[1].set_title("same color")
plt.subplots_adjust(top=0.8)
plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_23_0.png)



> ✅개선<br>
- 주별 평균 데이터를 사용해서, 이상치를 제거하는 효과를 줌
- 동일한 색상을 사용하면 사용자는 색상에 주의를 기울이지 않고 데이터 패턴에 주의를 기울일 수 있음

### 월별 판매량 (Violin Plot)


```python
weekly_group_df['month'] = weekly_group_df['date'].dt.month

fig, axes = plt.subplots(figsize=(14,6))
avg = weekly_group_df['sales'].mean()
sns.violinplot(y='sales', x='month', data=weekly_group_df, color='#B0E0E6')
plt.axhline(avg, ls='--', color='r', label=f'Sales Mean Line ({round(avg,2)})')
fig.suptitle('Month-wise Plot', fontweight='bold')
plt.legend(bbox_to_anchor=(0.98, 1.1),loc='upper right')
plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_26_0.png)



### 요일별 판매량 (Bar Plot)


```python
daily_group_df = train_df.groupby([pd.Grouper(key='date', freq='d')]).agg(sales = ('sales', 'mean')).reset_index()

order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

fig, ax = plt.subplots(figsize=(7,5))
palette = ['#dddddd'] * 7
palette[5:] = ['#00A000','#00A000']

daily_group_df['day_of_week'] = daily_group_df['date'].dt.day_name()
sns.barplot(x='sales', y='day_of_week', data=daily_group_df, order=order,
            errorbar=('ci', False), palette=palette, alpha=0.8)
ax.bar_label(ax.containers[0],fmt='%.0f')
sns.despine(left=True, bottom=True)
plt.xticks([])
plt.ylabel('')
plt.xlabel('')
fig.suptitle('Day of Week Sales Plot', fontweight='bold')
plt.tight_layout()
plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_28_0.png)



### 제품군 종류 (Bar Plot)

> 🧑: 어떤 제품군들이 판매되고 있는지도 알아보자

> ⚡ 참고할 시각화 규칙
1. **그래프 배치 순서에 의미를 두자!**
2. **색은 강조하고 싶은 부분에만! 최대한 적은 개수를 사용하자!**




```python
num_unique_family = len(train_df['family'].unique())
# 데이터를 미리 그룹화 해주었습니다. 미리 그룹화 하지 않으면 bar 그래프를 그리는데 오래 걸립니다.
family_group = train_df.groupby('family').agg(sales=('sales','mean')).reset_index()
sorted_family_group = train_df.groupby('family').agg(sales=('sales','mean')).reset_index().sort_values(by='sales', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 8))

for ax, data, title in zip(axes, [family_group, sorted_family_group],['Unsorted','Sorted']):
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(title)
    sns.barplot(y='family', x='sales', data=data, ax=ax)

plt.tight_layout()
fig.suptitle("Product Category(family)", y=1.1, fontsize=18, fontweight='bold')
fig.text(x=0.41,y= 1.045, s= f"sell {num_unique_family} product categories", fontsize=13)
plt.show()

```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_31_0.png)



> :☝ 해석
- 총 33개의 제품이 판매되고 있다.
- GROCERY1 제품이 가장 많이 판매되는 것도 확인할 수 있다.

> ❎ 문제점(첫번째 그림)<br>
그런데 그래프에 문제가 있다. 정렬이 되어있지 않아 데이터가 눈에 잘 들어오지 않는다.<br>
- 배치가 올바르지 않으면 그래프를 읽기 힘들다.
- 불필요하게 너무 많은 색을 사용하였다.


> ✅개선(두번째 그림)
- 그래프 배치에 신경을 쓰니, 어떤 제품들 순서로 많이 판매되는지를 알 수 있습니다. <br>

> 🧑: 이것만으로도 충분할수 있지만, 색이 불필요하게 많이 사용되었다.<br>
들어오는 정보가 많으면, 인간의 뇌는 정작 필요한 정보는 쉽게 놓친다고 한다.


```python
num_unique_family = len(train_df['family'].unique())
# 데이터를 미리 그룹화 해주었습니다. 미리 그룹화 하지 않으면 bar 그래프를 그리는데 오래 걸립니다.
family_group = train_df.groupby('family').agg(sales=('sales','mean')).reset_index().sort_values(by='sales', ascending=False)

#color map
color_map = ['#d4dddd' for _ in range(num_unique_family)]
color_map[:5] = ['#87CEEB']*5
color_map[:2] = ['#4682B4']*2

fig, ax = plt.subplots(figsize=(5, 8))
ax.tick_params(axis='y', labelsize=8)
sns.barplot(y='family', x='sales', data=family_group, palette=color_map)
fig.suptitle("Product Category(family)", x=0.3 )
fig.text(x=0.3, y= 0.90, s= f"sell {num_unique_family} product categories")
plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_34_0.png)



> ✅개선
- 많이 판매되는 5개 제품군을 강조할 수 있다.

### 주요 제품군들의 판매량

> 🧑: 주요 제품군별로 판매량 추세를 살펴보자

> ⚡ 참고할 시각화 규칙
> 1. 강조하고 싶은 데이터에만 색을 쓰자


```python
# 주요 판매 제품군
top_products = family_group[:5]['family']

weekly_family_group_df = train_df.groupby([pd.Grouper(key='date', freq='W'), 'family']).agg(sales = ('sales', 'mean'), onpromotions=('onpromotion', 'mean')).reset_index()
weekly_family_group_df=weekly_family_group_df[weekly_family_group_df['family'].isin(top_products)]
```


```python
fig, ax = plt.subplots(figsize=(12, 5))
# line plot
sns.lineplot(x='date', y='sales', hue='family',data=weekly_family_group_df)
fig.suptitle('Top 5 Family Sales Over Time', fontweight='bold')

plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_40_0.png)



>☝ 해석
- PRODUCE, BEVERAGES가 상당히 유사한 판매 패턴을 보인다.
- GROCERY 1도 판매량의 변화 패턴이 PRODUCE, BEVERAGES와 약간 유사해 보인다.



> ❎문제점<br>
- 너무 많은 색이 있어 패턴이 비슷한 데이터가 눈에 들어오지 않는다.

> 🧑: 패턴이 유사한 데이터를 강조 해보자.


```python
color_map = ['#d4dddd' for _ in range(5)]
color_map[0] = '#3498db'
color_map[3] = '#5d9cec'
color_map[4] = '#4b77be'


weekly_family_group_df = train_df.groupby([pd.Grouper(key='date', freq='W'), 'family']).agg(sales = ('sales', 'mean'), onpromotions=('onpromotion', 'mean')).reset_index()
weekly_family_group_df=weekly_family_group_df[weekly_family_group_df['family'].isin(top_products)]

fig, ax = plt.subplots(figsize=(12, 5))
# line plot
sns.lineplot(x='date', y='sales', hue='family',data=weekly_family_group_df, palette=color_map)
fig.suptitle('GROCERY I,BEVERAGES,PRODUCE', fontweight='bold')

plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_43_0.png)



> ✅개선<br>
- 유사한 패턴을 서로 유사한 계열의 색으로 표현하여, 유사성이 더 눈에 띈다.

### Promotions

> 🧑: 프로모션 트렌드를 살펴보자.


```python
fig, ax = plt.subplots(figsize=(12, 5))
# line plot
sns.lineplot(x='date', y='onpromotions',data=weekly_family_group_df,  color='#808080')
fig.suptitle('Promotions Over Time', fontweight='bold')

plt.show()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_47_0.png)



>☝ 해석
- promotion이 점점 증가하는 경향이 보인다.


### 프로모션과 판매량 상관관계

> 🧑: 프로모션이 많으면, 제품 판매량이 증가할까?


```python
# 데이터가 정상성을 띄는 2016년 이후만 활용
promotion_df = train_df[train_df['date'] >= '2016-01-01']
```


```python
# family = promotion_df.family.unique()
promotional_products = ['PLAYERS AND ELECTRONICS', 'PRODUCE',
                        'SCHOOL AND OFFICE SUPPLIES', 'DELI',
                        'BEAUTY']
```


```python
scatter_kws = {'color': 'blue', 'alpha': 0.4, 's': 15}
line_kws = {'color': 'red'}

fig, axes = plt.subplots(2,3, figsize=(15,8))

for ax, p in zip(axes.flatten(), promotional_products):
    sns.regplot(data=promotion_df[promotion_df['family']==p],
               x='onpromotion',y='sales', ax=ax,scatter_kws=scatter_kws, line_kws=line_kws)
    ax.set_title(f'{p}')
fig.suptitle("linear relationship with sales")
plt.tight_layout()
plt.show()

```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_52_0.png)



> 🧑: 몇몇 제품군은 Promotion과 Sales에 양의 선형상관관계가 있어 보인다.

### ACF/PACF 상관도표(Correlogram)

> 🧑: ACF/PACF를 사용해 데이터의 정상성을 평가하자(이미 판매량이 점점 증가하는 추세가 있음을 확인했지만). <br>
> - ACF(자기상관함수)는 현재값과 특정 시점 이전(lag)의 값의 상관관계를 보여준다. <br>
> - PACF는 다른 시차의 영향을 제외하고, 현재값과 특정시점 이전의 값의 상관관계를 보여준다. <br>
> - 상관도표를 통해 한눈에 시계열 데이터의 정상성을 판단할 수 있다. <br>
> - 또한 ARIMA 모형의 차수를 결정하는데 도움이 된다.




```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```


```python
fig, axes = plt.subplots(5,2, figsize=(30,20))

for (ax, k) in zip(axes, top_products):
    d = train_df[train_df['family']==k]
    d = d.groupby(by='date')['sales'].mean()
    plot_acf(d,lags=300, alpha=0.05, ax=ax[0])
    plot_pacf(d, lags=300, alpha=0.05, ax=ax[1])
    ax[0].set_title(f"{k} Autocorrelation")
    ax[1].set_title(f"{k} ParticialAutocorrelation")
    plt.tight_layout()
```



![png](/assets/images/contents/EDA/EDA-1_files/EDA-1_56_0.png)



> 🧑 :
> - ACF, PACF가 특정한 패턴이 없고 랜덤하면, 데이터가 Stationary(정상성)을 가진다고 할 수 있음
> - ACF, PACF가 LAG 1, LAG 2 이후 확 떨어지면 데이터가 Stationary(정상성)을 가진다고 할 수 있음
- ACF, PACF가 천천히 감소하는 형태면 전형적으로 Nonstational한 데이터임


>☝ 해석
- ACF, PACF가 천천히 감소하는 형태로, 데이터가 비정상적임
- Lag 7을 주기로 강한 상관관계가 나타남


