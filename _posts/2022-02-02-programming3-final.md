```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```

### 3번
(60점) 주어진 2015년~2020년까지 6개 연도별 서울 아파트 실거래가 데이터와 아파트정보 데이터를 이용해서 다음 문제를 해결하는 파이썬 코드를 작성하시오.

실거래가 파일:
DealYear, DealMonth, DealDay
거래 연월일
AptCode
아파트 식별 코드값
DealAmount
거래금액(단위: 만원, 천 단위로 콤마(,)가 표시된 문자열 데이터)
Dong
서울시 동명

아파트데이터 파일:
AptCode
아파트 식별 코드값
ApartmentName
아파트명
AreaforExclusiveUse
전용면적(m2)
BuildYear
건축연도
Floor
층수
RegionalCode
서울시 구 코드
CancelDealType
거래 취소 신고 여부 (O: 거래취소신고)
CancelDealDay
거래 취소 신고일

(1) (10점) 연도별 실거래데이터를 연결하여 하나의 DataFrame으로 만들고, DataFrame의 인덱스를 거래일자의 datetime 값이 되도록 하시오. 그리고, DealAmount 컬럼은 정수형 타입으로 변환하시오.
이하 문제는 (1)에서 연결한 전체 데이터를 이용한다. (그러나, 연결하지 못한 경우에는 가장 마지막 데이터인 “실거래가_202012.csv” 파일 1개를 이용해도 감점은 없음)

(2) (10점) 전체 데이터에서 취소 건수의 개수를 구하고, 최초로 취소 신고가 있었던 날짜를 구하시오.

(3) (10점) 데이터에서 취소 거래 데이터를 제거하시오. (거래 취소 신고 데이터와 바로 직전에 있는 거래 신고 데이터를 동시에 제거시켜야 함)
이하 문제는 (3)에서 취소 거래를 삭제한 결과 데이터를 이용한다. (그러나 삭제하지 못한 데이터를 이용하여도 감점은 없음)

(4) (10점) 전체 거래에 대해 구별로 평균 거래가격이 가장 높은 동과 해당 동의 평균 매매가격을 구하시오.
RegionalCode Dong AvgPrice
xxxxx xxxxx xxxxx
xxxxx xxxxx xxxxx
xxxxx xxxxx xxxxx

(5) (10점) 연도별/구별로 m2 당 거래가격의 평균을 구하고 각 연도별로 평균가격의 상승률이 가장 높은 구를 구하시오.
Year RegionalCode AvgPrice
xxxxx xxxxx xxxxx
xxxxx xxxxx xxxxx
xxxxx xxxxx xxxxx

Year Highest_Region
xxxxx xxxxx
xxxxx xxxxx

(6) (10점) 전체 거래에 대해서 DealYear와 BuildYear의 차이(=Years)를 구하고 0~5년, 6년~10년, 11년~15년, 16년~20년, 21년~30년 이상, 31년 이상으로 구분하여, 분기별 / Years 구간별로 거개건수와 m2당 거래가격의 평균을 구하시오.

Quarter Counts AvgPrice 0-5 6-10 … 30- 0-5 6-10 … 30-
2015Q1
xxxxx
xxxxx
…
xxxxx
xxxxx
xxxxx
…
xxxxx
2015Q2
xxxxx
xxxxx
…
xxxxx
xxxxx
xxxxx
…
xxxxx
…
…
…
…
…
…
…
…
…
2020Q4
xxxxx
xxxxx
…
xxxxx
xxxxx
xxxxx
…
xxxxx


```python
# (1)
```


```python
import os
folders=os.listdir('./')
folders
```




    ['실거래가_201501.csv',
     '실거래가_201502.csv',
     '실거래가_201503.csv',
     '실거래가_201504.csv',
     '실거래가_201505.csv',
     '실거래가_201506.csv',
     '실거래가_201507.csv',
     '실거래가_201508.csv',
     '실거래가_201509.csv',
     '실거래가_201510.csv',
     '실거래가_201511.csv',
     '실거래가_201512.csv',
     '실거래가_201601.csv',
     '실거래가_201602.csv',
     '실거래가_201603.csv',
     '실거래가_201604.csv',
     '실거래가_201605.csv',
     '실거래가_201606.csv',
     '실거래가_201607.csv',
     '실거래가_201608.csv',
     '실거래가_201609.csv',
     '실거래가_201610.csv',
     '실거래가_201611.csv',
     '실거래가_201612.csv',
     '실거래가_201701.csv',
     '실거래가_201702.csv',
     '실거래가_201703.csv',
     '실거래가_201704.csv',
     '실거래가_201705.csv',
     '실거래가_201706.csv',
     '실거래가_201707.csv',
     '실거래가_201708.csv',
     '실거래가_201709.csv',
     '실거래가_201710.csv',
     '실거래가_201711.csv',
     '실거래가_201712.csv',
     '실거래가_201801.csv',
     '실거래가_201802.csv',
     '실거래가_201803.csv',
     '실거래가_201804.csv',
     '실거래가_201805.csv',
     '실거래가_201806.csv',
     '실거래가_201807.csv',
     '실거래가_201808.csv',
     '실거래가_201809.csv',
     '실거래가_201810.csv',
     '실거래가_201811.csv',
     '실거래가_201812.csv',
     '실거래가_201901.csv',
     '실거래가_201902.csv',
     '실거래가_201903.csv',
     '실거래가_201904.csv',
     '실거래가_201905.csv',
     '실거래가_201906.csv',
     '실거래가_201907.csv',
     '실거래가_201908.csv',
     '실거래가_201909.csv',
     '실거래가_201910.csv',
     '실거래가_201911.csv',
     '실거래가_201912.csv',
     '실거래가_202001.csv',
     '실거래가_202002.csv',
     '실거래가_202003.csv',
     '실거래가_202004.csv',
     '실거래가_202005.csv',
     '실거래가_202006.csv',
     '실거래가_202007.csv',
     '실거래가_202008.csv',
     '실거래가_202009.csv',
     '실거래가_202010.csv',
     '실거래가_202011.csv',
     '실거래가_202012.csv']




```python
os.getcwd()
# os.chdir('./Codes/기말문제')
os.getcwd()
```




    'c:\\Users\\jihoonlee\\OneDrive\\바탕 화면\\kaist\\2021_Fall\\Programming3\\Codes\\기말문제'




```python
df_all=pd.DataFrame()

for files in folders:
    df=pd.read_csv(files, encoding='CP949')
    df_all=pd.concat([df_all,df])
df_all.head()
```




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
      <th>DealYear</th>
      <th>DealMonth</th>
      <th>DealDay</th>
      <th>DealAmount</th>
      <th>AptCode</th>
      <th>AreaforExclusiveUse</th>
      <th>Floor</th>
      <th>CancelDealType</th>
      <th>CancelDealDay</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>69,700</td>
      <td>C0068</td>
      <td>84.93</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>43,500</td>
      <td>D0013</td>
      <td>84.93</td>
      <td>13</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>36,900</td>
      <td>F0013</td>
      <td>84.95</td>
      <td>23</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>26,500</td>
      <td>F0071</td>
      <td>60.48</td>
      <td>12</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>33,000</td>
      <td>G0001</td>
      <td>84.77</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>




```python
from datetime import datetime
```


```python
df_all['Date'] = df_all['DealYear'].astype(str) + '-' + df_all['DealMonth'].astype(str) + '-' + df_all['DealDay'].astype(str) # 일별 데이터
df_all['Date'] = pd.to_datetime(df_all['Date'])
df_all['Date']
df_all.set_index('Date', inplace=True)
df_all
```




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
      <th>DealYear</th>
      <th>DealMonth</th>
      <th>DealDay</th>
      <th>DealAmount</th>
      <th>AptCode</th>
      <th>AreaforExclusiveUse</th>
      <th>Floor</th>
      <th>CancelDealType</th>
      <th>CancelDealDay</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>69,700</td>
      <td>C0068</td>
      <td>84.93</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>43,500</td>
      <td>D0013</td>
      <td>84.93</td>
      <td>13</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>36,900</td>
      <td>F0013</td>
      <td>84.95</td>
      <td>23</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>26,500</td>
      <td>F0071</td>
      <td>60.48</td>
      <td>12</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>33,000</td>
      <td>G0001</td>
      <td>84.77</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>37,800</td>
      <td>Y0122</td>
      <td>27.61</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>111,000</td>
      <td>Y0166</td>
      <td>84.74</td>
      <td>12</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>104,000</td>
      <td>Y0171</td>
      <td>84.74</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>79,000</td>
      <td>Y0351</td>
      <td>84.50</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>157,000</td>
      <td>Y0455</td>
      <td>84.94</td>
      <td>32</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>576324 rows × 9 columns</p>
</div>




```python
df_all.DealAmount = df_all.DealAmount.str.replace(',','').astype('int64')
```


```python
df_all.DealAmount
```




    Date
    2015-01-01     69700
    2015-01-01     43500
    2015-01-01     36900
    2015-01-01     26500
    2015-01-01     33000
                   ...  
    2020-12-31     37800
    2020-12-31    111000
    2020-12-31    104000
    2020-12-31     79000
    2020-12-31    157000
    Name: DealAmount, Length: 576324, dtype: int64




```python
# (2)
df_cancel = df_all[df_all.CancelDealType=='O'].copy()
df_cancel.CancelDealType = 1
df_cancel.CancelDealType.sum()
```




    3037




```python
df_cancel.CancelDealDay.head(1)
```




    Date
    2020-02-21    20.03.16
    Name: CancelDealDay, dtype: object




```python
# (3)
df_all
```




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
      <th>DealYear</th>
      <th>DealMonth</th>
      <th>DealDay</th>
      <th>DealAmount</th>
      <th>AptCode</th>
      <th>AreaforExclusiveUse</th>
      <th>Floor</th>
      <th>CancelDealType</th>
      <th>CancelDealDay</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>69700</td>
      <td>C0068</td>
      <td>84.93</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>43500</td>
      <td>D0013</td>
      <td>84.93</td>
      <td>13</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>36900</td>
      <td>F0013</td>
      <td>84.95</td>
      <td>23</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>26500</td>
      <td>F0071</td>
      <td>60.48</td>
      <td>12</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2015-01-01</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>33000</td>
      <td>G0001</td>
      <td>84.77</td>
      <td>19</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>37800</td>
      <td>Y0122</td>
      <td>27.61</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>111000</td>
      <td>Y0166</td>
      <td>84.74</td>
      <td>12</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>104000</td>
      <td>Y0171</td>
      <td>84.74</td>
      <td>5</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>79000</td>
      <td>Y0351</td>
      <td>84.50</td>
      <td>7</td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <th>2020-12-31</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>157000</td>
      <td>Y0455</td>
      <td>84.94</td>
      <td>32</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>
<p>576324 rows × 9 columns</p>
</div>




```python
# (4)
region=pd.read_csv('C:/Users/jihoonlee/OneDrive/바탕 화면/kaist/2021_Fall/Programming3/Codes/아파트데이터.csv', encoding='CP949')
region.columns=["AptCode", "Dong", "RegionalName", "BuildYear", "RegionalCode"]
region
```




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
      <th>AptCode</th>
      <th>Dong</th>
      <th>RegionalName</th>
      <th>BuildYear</th>
      <th>RegionalCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>X0390</td>
      <td>가락동</td>
      <td>e지브로</td>
      <td>2004</td>
      <td>11710</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X0069</td>
      <td>가락동</td>
      <td>sk파크타워</td>
      <td>2003</td>
      <td>11710</td>
    </tr>
    <tr>
      <th>2</th>
      <td>X0052</td>
      <td>가락동</td>
      <td>가락(1차)쌍용아파트</td>
      <td>1997</td>
      <td>11710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>X0064</td>
      <td>가락동</td>
      <td>가락3차쌍용스윗닷홈 101동, 102동</td>
      <td>2005</td>
      <td>11710</td>
    </tr>
    <tr>
      <th>4</th>
      <td>X0154</td>
      <td>가락동</td>
      <td>가락3차쌍용스윗닷홈(103동)</td>
      <td>2005</td>
      <td>11710</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8694</th>
      <td>B0019</td>
      <td>흥인동</td>
      <td>aaa</td>
      <td>1994</td>
      <td>11140</td>
    </tr>
    <tr>
      <th>8695</th>
      <td>B0018</td>
      <td>흥인동</td>
      <td>동대문 와이즈 캐슬</td>
      <td>2014</td>
      <td>11140</td>
    </tr>
    <tr>
      <th>8696</th>
      <td>B0103</td>
      <td>흥인동</td>
      <td>동대문솔하임</td>
      <td>2016</td>
      <td>11140</td>
    </tr>
    <tr>
      <th>8697</th>
      <td>B0109</td>
      <td>흥인동</td>
      <td>위더스하임</td>
      <td>2018</td>
      <td>11140</td>
    </tr>
    <tr>
      <th>8698</th>
      <td>B0066</td>
      <td>흥인동</td>
      <td>청계천 두산위브더제니스</td>
      <td>2014</td>
      <td>11140</td>
    </tr>
  </tbody>
</table>
<p>8699 rows × 5 columns</p>
</div>




```python
# df_all = pd.merge(df_all, region, how='left', on='AptCode')
df_all.iloc[:,-8:]

```




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
      <th>DealYear</th>
      <th>DealMonth</th>
      <th>DealDay</th>
      <th>DealAmount</th>
      <th>AptCode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>69700</td>
      <td>C0068</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>43500</td>
      <td>D0013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>36900</td>
      <td>F0013</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>26500</td>
      <td>F0071</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>33000</td>
      <td>G0001</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>576319</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>37800</td>
      <td>Y0122</td>
    </tr>
    <tr>
      <th>576320</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>111000</td>
      <td>Y0166</td>
    </tr>
    <tr>
      <th>576321</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>104000</td>
      <td>Y0171</td>
    </tr>
    <tr>
      <th>576322</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>79000</td>
      <td>Y0351</td>
    </tr>
    <tr>
      <th>576323</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>157000</td>
      <td>Y0455</td>
    </tr>
  </tbody>
</table>
<p>576324 rows × 5 columns</p>
</div>




```python
df_all.groupby('Dong')['DealAmount'].mean().sort_values(ascending=False).head(1)
```




    Dong
     신문로2가    275500.0
    Name: DealAmount, dtype: float64




```python
# (5)
df_all['AvgPrice'] = df_all['DealAmount']/df_all['AreaforExclusiveUse'] # 평당 아파트 가격
df_all['AvgPrice']
```




    0          820.675851
    1          512.186507
    2          434.373161
    3          438.161376
    4          389.288663
                 ...     
    576319    1369.069178
    576320    1309.889072
    576321    1227.283455
    576322     934.911243
    576323    1848.363551
    Name: AvgPrice, Length: 576324, dtype: float64




```python
df_price = df_all.groupby(['DealYear','RegionalCode'])[['AvgPrice']].mean()
df_price
```




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
      <th></th>
      <th>AvgPrice</th>
    </tr>
    <tr>
      <th>DealYear</th>
      <th>RegionalCode</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">2015</th>
      <th>11110</th>
      <td>603.301950</td>
    </tr>
    <tr>
      <th>11140</th>
      <td>639.877588</td>
    </tr>
    <tr>
      <th>11170</th>
      <td>819.798573</td>
    </tr>
    <tr>
      <th>11200</th>
      <td>657.019538</td>
    </tr>
    <tr>
      <th>11215</th>
      <td>687.111999</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2020</th>
      <th>11620</th>
      <td>871.085534</td>
    </tr>
    <tr>
      <th>11650</th>
      <td>1913.571563</td>
    </tr>
    <tr>
      <th>11680</th>
      <td>2086.317298</td>
    </tr>
    <tr>
      <th>11710</th>
      <td>1601.218608</td>
    </tr>
    <tr>
      <th>11740</th>
      <td>1169.695376</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 1 columns</p>
</div>




```python
df_ret = df_all.groupby(['RegionalCode','DealYear'])[['AvgPrice']].mean().pct_change().unstack().T.fillna(0)
df_ret
```




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
      <th>RegionalCode</th>
      <th>11110</th>
      <th>11140</th>
      <th>11170</th>
      <th>11200</th>
      <th>11215</th>
      <th>11230</th>
      <th>11260</th>
      <th>11290</th>
      <th>11305</th>
      <th>11320</th>
      <th>...</th>
      <th>11500</th>
      <th>11530</th>
      <th>11545</th>
      <th>11560</th>
      <th>11590</th>
      <th>11620</th>
      <th>11650</th>
      <th>11680</th>
      <th>11710</th>
      <th>11740</th>
    </tr>
    <tr>
      <th></th>
      <th>DealYear</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">AvgPrice</th>
      <th>2015</th>
      <td>0.000000</td>
      <td>-0.381595</td>
      <td>-0.278149</td>
      <td>-0.588004</td>
      <td>-0.532945</td>
      <td>-0.619927</td>
      <td>-0.553063</td>
      <td>-0.345434</td>
      <td>-0.490234</td>
      <td>-0.484248</td>
      <td>...</td>
      <td>-0.560655</td>
      <td>-0.521595</td>
      <td>-0.415806</td>
      <td>-0.232000</td>
      <td>-0.458991</td>
      <td>-0.597217</td>
      <td>0.120571</td>
      <td>-0.403811</td>
      <td>-0.587471</td>
      <td>-0.603777</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>0.074957</td>
      <td>0.065964</td>
      <td>0.110650</td>
      <td>0.094998</td>
      <td>0.041267</td>
      <td>0.127495</td>
      <td>0.059493</td>
      <td>0.082711</td>
      <td>0.030104</td>
      <td>0.086313</td>
      <td>...</td>
      <td>0.156556</td>
      <td>0.091482</td>
      <td>-0.004855</td>
      <td>0.104162</td>
      <td>0.067677</td>
      <td>0.094826</td>
      <td>0.131327</td>
      <td>0.122343</td>
      <td>0.082792</td>
      <td>0.088333</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>0.074125</td>
      <td>0.115922</td>
      <td>0.112170</td>
      <td>0.152593</td>
      <td>0.105743</td>
      <td>0.102910</td>
      <td>0.057033</td>
      <td>0.071235</td>
      <td>0.071069</td>
      <td>0.085253</td>
      <td>...</td>
      <td>0.122737</td>
      <td>0.079535</td>
      <td>0.058744</td>
      <td>0.144466</td>
      <td>0.094746</td>
      <td>0.054848</td>
      <td>0.121265</td>
      <td>0.144996</td>
      <td>0.105150</td>
      <td>0.081241</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>0.097248</td>
      <td>0.145111</td>
      <td>0.183880</td>
      <td>0.264808</td>
      <td>0.196349</td>
      <td>0.135106</td>
      <td>0.086166</td>
      <td>0.091594</td>
      <td>0.095263</td>
      <td>0.075673</td>
      <td>...</td>
      <td>0.117576</td>
      <td>0.094893</td>
      <td>0.089209</td>
      <td>0.177524</td>
      <td>0.178557</td>
      <td>0.121640</td>
      <td>0.180371</td>
      <td>0.161251</td>
      <td>0.168109</td>
      <td>0.149757</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>0.329189</td>
      <td>0.175444</td>
      <td>0.220806</td>
      <td>0.187380</td>
      <td>0.168372</td>
      <td>0.213266</td>
      <td>0.221136</td>
      <td>0.225077</td>
      <td>0.180724</td>
      <td>0.177884</td>
      <td>...</td>
      <td>0.169684</td>
      <td>0.188348</td>
      <td>0.199824</td>
      <td>0.229900</td>
      <td>0.212640</td>
      <td>0.180190</td>
      <td>0.201167</td>
      <td>0.205212</td>
      <td>0.197504</td>
      <td>0.173542</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>0.018478</td>
      <td>0.108503</td>
      <td>0.089621</td>
      <td>0.181347</td>
      <td>0.168311</td>
      <td>0.183759</td>
      <td>0.149821</td>
      <td>0.185609</td>
      <td>0.159158</td>
      <td>0.136475</td>
      <td>...</td>
      <td>0.131368</td>
      <td>0.088026</td>
      <td>0.194184</td>
      <td>0.135482</td>
      <td>0.154018</td>
      <td>0.149383</td>
      <td>0.089999</td>
      <td>0.016793</td>
      <td>0.111448</td>
      <td>0.161163</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 25 columns</p>
</div>




```python
# (6)
df_all['Years'] = df_all.DealYear - df_all.BuildYear
df_all.Years
```




    2015Q1    23
    2015Q1    11
    2015Q1    15
    2015Q1    26
    2015Q1    20
              ..
    2020Q4    33
    2020Q4    11
    2020Q4    11
    2020Q4    16
    2020Q4     4
    Freq: Q-DEC, Name: Years, Length: 576324, dtype: int64




```python
df_all.DealYear
df_all.DealMonth
index = pd.PeriodIndex(year=df_all.DealYear, month=df_all.DealMonth,
                       freq='Q-DEC')
index
df_all.index = index
```


```python
df_all
```




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
      <th>DealYear</th>
      <th>DealMonth</th>
      <th>DealDay</th>
      <th>DealAmount</th>
      <th>AptCode</th>
      <th>AreaforExclusiveUse</th>
      <th>Floor</th>
      <th>CancelDealType</th>
      <th>CancelDealDay</th>
      <th>Dong</th>
      <th>RegionalName</th>
      <th>BuildYear</th>
      <th>RegionalCode</th>
      <th>Price/SquareMeter</th>
      <th>AvgPrice</th>
      <th>Years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015Q1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>69700</td>
      <td>C0068</td>
      <td>84.93</td>
      <td>5</td>
      <td></td>
      <td></td>
      <td>보광동</td>
      <td>신동아1</td>
      <td>1992</td>
      <td>11170</td>
      <td>820.675851</td>
      <td>820.675851</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2015Q1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>43500</td>
      <td>D0013</td>
      <td>84.93</td>
      <td>13</td>
      <td></td>
      <td></td>
      <td>마장동</td>
      <td>신성미소지움</td>
      <td>2004</td>
      <td>11200</td>
      <td>512.186507</td>
      <td>512.186507</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2015Q1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>36900</td>
      <td>F0013</td>
      <td>84.95</td>
      <td>23</td>
      <td></td>
      <td></td>
      <td>전농동</td>
      <td>전농SK</td>
      <td>2000</td>
      <td>11230</td>
      <td>434.373161</td>
      <td>434.373161</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2015Q1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>26500</td>
      <td>F0071</td>
      <td>60.48</td>
      <td>12</td>
      <td></td>
      <td></td>
      <td>회기동</td>
      <td>신현대</td>
      <td>1989</td>
      <td>11230</td>
      <td>438.161376</td>
      <td>438.161376</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2015Q1</th>
      <td>2015</td>
      <td>1</td>
      <td>1</td>
      <td>33000</td>
      <td>G0001</td>
      <td>84.77</td>
      <td>19</td>
      <td></td>
      <td></td>
      <td>면목동</td>
      <td>두산2</td>
      <td>1995</td>
      <td>11260</td>
      <td>389.288663</td>
      <td>389.288663</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2020Q4</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>37800</td>
      <td>Y0122</td>
      <td>27.61</td>
      <td>5</td>
      <td></td>
      <td></td>
      <td>천호동</td>
      <td>금호아파트</td>
      <td>1987</td>
      <td>11740</td>
      <td>1369.069178</td>
      <td>1369.069178</td>
      <td>33</td>
    </tr>
    <tr>
      <th>2020Q4</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>111000</td>
      <td>Y0166</td>
      <td>84.74</td>
      <td>12</td>
      <td></td>
      <td></td>
      <td>강일동</td>
      <td>강일리버파크4단지</td>
      <td>2009</td>
      <td>11740</td>
      <td>1309.889072</td>
      <td>1309.889072</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020Q4</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>104000</td>
      <td>Y0171</td>
      <td>84.74</td>
      <td>5</td>
      <td></td>
      <td></td>
      <td>강일동</td>
      <td>강일리버파크9단지</td>
      <td>2009</td>
      <td>11740</td>
      <td>1227.283455</td>
      <td>1227.283455</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2020Q4</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>79000</td>
      <td>Y0351</td>
      <td>84.50</td>
      <td>7</td>
      <td></td>
      <td></td>
      <td>성내동</td>
      <td>우림루미아트</td>
      <td>2004</td>
      <td>11740</td>
      <td>934.911243</td>
      <td>934.911243</td>
      <td>16</td>
    </tr>
    <tr>
      <th>2020Q4</th>
      <td>2020</td>
      <td>12</td>
      <td>31</td>
      <td>157000</td>
      <td>Y0455</td>
      <td>84.94</td>
      <td>32</td>
      <td></td>
      <td></td>
      <td>고덕동</td>
      <td>고덕래미안 힐스테이트아파트</td>
      <td>2016</td>
      <td>11740</td>
      <td>1848.363551</td>
      <td>1848.363551</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>576324 rows × 16 columns</p>
</div>




```python
bins = [0, 6, 11, 16, 21, 31, 100]
```


```python
bins_label = ['0-5','6-10','11-15','16-20','21-30','31년 이상']
```


```python
df_all['Counts'] = pd.cut(df_all['Years'], bins, right=True, labels=bins_label)
df_all['Counts']
```




    2015Q1     21-30
    2015Q1      6-10
    2015Q1     11-15
    2015Q1     21-30
    2015Q1     16-20
               ...  
    2020Q4    31년 이상
    2020Q4      6-10
    2020Q4      6-10
    2020Q4     11-15
    2020Q4       0-5
    Freq: Q-DEC, Name: Counts, Length: 576324, dtype: category
    Categories (6, object): ['0-5' < '6-10' < '11-15' < '16-20' < '21-30' < '31년 이상']



### 2번
(30점) (for 또는 while 과 같은 반복문을 사용하지 말 것) A씨는 향후 1년간 다음의 수익률 분포를 가지는 두 가지 투자안 (1)과 (2)에 대해 적정 비율로 배분하여 투자하고자 한다.

◼ 투자안 (1)의 연속복리수익률: 𝑟1 (평균이 6%, 표준편차가 1%)

◼ 투자안 (2)의 연속복리수익률: 𝑟2 (평균이 10%, 표준편차가 50%)

◼ 𝑟1과 𝑟2 의 상관계수: 30%
투자안 (1)에 𝛼, (2)에 1−𝛼의 비율로 투자할 때 (0<𝛼<1), 1년 후 자산(wealth)은 다음과 같다. 𝑊=𝛼×𝑒𝑥𝑝(𝑟1)+(1−𝛼)×exp (𝑟2)
A씨의 W에 대한 효용함수는 다음과 같고, 효용의 기대값(expectation)을 최대화하는 의사 결정을 한다. 𝑈(𝑊)=𝑊1−𝜌−11−𝜌
이 식에서 𝜌는 A씨의 위험회피성향을 나타내며, 0.9 라고 가정한다.

(1) (10점) 몬테카를로 시뮬레이션을 위해서 𝑟1과 𝑟2의 임의의 난수를 10,000회 발생시키고, 10,000개의 수익률에 대해서 𝑟1과 𝑟2의 평균과 표준편차(unbiased) 그리고 상관계수를 구하는 코드를 작성하시오.

(2) (10점) 위의 문제 (1)에서 생성한 수익률을 이용해서 효용의 기대값을 계산하는 함수를 만드시오. 시뮬레이션된 각 시나리오에 대해서 효용을 계산하고 계산된 효용 값들을 평균한다. (파라미터로 𝛼와 수익률 Array (10,000×2)를 입력받아 기대 효용을 리턴하는 함수임) 또한 이 함수를 이용해서 𝛼=0.4라고 가정하고, A씨의 기대 효용 값을 구하시오.

(3) (10점) A씨의 기대 효용의 값을 최대로 만드는 𝛼의 값을 구하시오. 단, 𝛼는 0이상 1 이하의 값이어야 한다. (scipy.optimize를 이용하시오.)


```python
# (1)
numSim = 10000
mu = np.array([0.06,0.1])
sigma = np.array([0.01,0.5])
```


```python
corr = np.array([[1, 0.3],
                 [0.3, 1]])
cov = corr*sigma*sigma.T # Broadcasting
```


```python
x = np.random.multivariate_normal(mu, cov, numSim)
r = np.exp(x)
r
```

    <ipython-input-45-8d157270f93e>:1: RuntimeWarning: covariance is not positive-semidefinite.
      x = np.random.multivariate_normal(mu, cov, numSim)
    




    array([[1.07119658, 1.46793998],
           [1.05189268, 0.58388602],
           [1.0586104 , 1.50322762],
           ...,
           [1.06372211, 0.79301197],
           [1.08714001, 0.98813846],
           [1.06799167, 0.93123876]])




```python
r.mean(0)
```


```python
r.std(0)
```




    array([0.00995055, 0.69423494])




```python
A = r[:,0]
B = r[:,1]
np.corrcoef(A,B)
```




    array([[1.        , 0.01686854],
           [0.01686854, 1.        ]])




```python
# (2)
alpha = 0.4
rho = 0.9
n=2
a0 = np.array([alpha, 1-alpha]).reshape(-1,1)
a0
```




    array([[0.4],
           [0.6]])




```python
def u(a,r): # expected utility function
    w = np.exp(r).dot(a) # wealth
    utility = ( w**(1-rho)-1 ) / (1-rho)
    return utility.mean()
```


```python
u(a0,r)
```




    1.3316514059083184




```python
# (3)
target_utiliy = np.linspace(0.4,2.4,201)
target_utiliy
```




    array([0.4 , 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 ,
           0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61,
           0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72,
           0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83,
           0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94,
           0.95, 0.96, 0.97, 0.98, 0.99, 1.  , 1.01, 1.02, 1.03, 1.04, 1.05,
           1.06, 1.07, 1.08, 1.09, 1.1 , 1.11, 1.12, 1.13, 1.14, 1.15, 1.16,
           1.17, 1.18, 1.19, 1.2 , 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27,
           1.28, 1.29, 1.3 , 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38,
           1.39, 1.4 , 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49,
           1.5 , 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6 ,
           1.61, 1.62, 1.63, 1.64, 1.65, 1.66, 1.67, 1.68, 1.69, 1.7 , 1.71,
           1.72, 1.73, 1.74, 1.75, 1.76, 1.77, 1.78, 1.79, 1.8 , 1.81, 1.82,
           1.83, 1.84, 1.85, 1.86, 1.87, 1.88, 1.89, 1.9 , 1.91, 1.92, 1.93,
           1.94, 1.95, 1.96, 1.97, 1.98, 1.99, 2.  , 2.01, 2.02, 2.03, 2.04,
           2.05, 2.06, 2.07, 2.08, 2.09, 2.1 , 2.11, 2.12, 2.13, 2.14, 2.15,
           2.16, 2.17, 2.18, 2.19, 2.2 , 2.21, 2.22, 2.23, 2.24, 2.25, 2.26,
           2.27, 2.28, 2.29, 2.3 , 2.31, 2.32, 2.33, 2.34, 2.35, 2.36, 2.37,
           2.38, 2.39, 2.4 ])




```python
for u in target_utiliy:
    c = ({"type":"eq", "fun": lambda x: x.sum()-1},
         {"type":"eq", "fun": lambda x: u(x,r)-r},
         {"type":"ineq", "fun": lambda x: x},
         {"type":"ineq", "fun": lambda x: 1-x})
    res = maximize(u, a0, )
```

### 1번
(10점) 영어 알파벳 대문자와 소문자 총 52개 중에 중복되지 않는 n개로 구성된 임의의 문자열을 만드는 함수를 작성하시오.

◼ 1보다 큰 정수 n을 함수의 파라미터로 받는 함수임

◼ 파이썬 내장함수인 chr(x)과 ord(x) 함수를 이용하시오.

◼ 함수를 이용해서 n=2 부터 10까지 반복하면서 생성된 문자열을 리스트로 만들고 화면에 출력하시오.


```python
chr(65) # A
chr(122) # z
```




    'z'




```python
vchr = np.vectorize(chr)
vord = np.vectorize(ord)
```


```python
from string import ascii_lowercase, ascii_uppercase
alpha = list(ascii_lowercase + ascii_uppercase)
alpha = vord(alpha) # ord
alpha
```




    array([ 97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
           110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
            65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
            78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90])




```python
def fun(n):
    result = ''.join(vchr(np.random.choice(alpha,n,replace=False))) # chr
    return result
```


```python
l = []
for i in range(2,11):
    l.append(fun(i))
l
```




    ['VN',
     'Rdr',
     'AxBo',
     'yltpx',
     'fTdKGv',
     'AnEsayj',
     'xqsQpOKS',
     'CgAsfqkyh',
     'VMZDizWxlq']


