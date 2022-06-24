```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
plt.style.use('seaborn')
%matplotlib inline
```


```python
pwd

```




    '/Users/adeekshyagurung/Documents/My First python Project/MyNew JN Folder'




```python
confirmed_cases = pd.read_csv("/Users/adeekshyagurung/Downloads/Coronavirus prediction analysis/time_series_covid-19_confirmed.csv")
```


```python
deaths_reported = pd.read_csv("/Users/adeekshyagurung/Downloads/Coronavirus prediction analysis/time_series_covid-19_deaths.csv")
```


```python
recovered_cases = pd.read_csv("/Users/adeekshyagurung/Downloads/Coronavirus prediction analysis/time_series_covid-19_recovered.csv")
```


```python
#Displaying the dataset
confirmed_cases.head()

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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>3/6/20</th>
      <th>3/7/20</th>
      <th>3/8/20</th>
      <th>3/9/20</th>
      <th>3/10/20</th>
      <th>3/11/20</th>
      <th>3/12/20</th>
      <th>3/13/20</th>
      <th>3/14/20</th>
      <th>3/15/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Thailand</td>
      <td>15.0000</td>
      <td>101.0000</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>7</td>
      <td>8</td>
      <td>8</td>
      <td>...</td>
      <td>48</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>53</td>
      <td>59</td>
      <td>70</td>
      <td>75</td>
      <td>82</td>
      <td>114</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Japan</td>
      <td>36.0000</td>
      <td>138.0000</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>420</td>
      <td>461</td>
      <td>502</td>
      <td>511</td>
      <td>581</td>
      <td>639</td>
      <td>639</td>
      <td>701</td>
      <td>773</td>
      <td>839</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Singapore</td>
      <td>1.2833</td>
      <td>103.8333</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>...</td>
      <td>130</td>
      <td>138</td>
      <td>150</td>
      <td>150</td>
      <td>160</td>
      <td>178</td>
      <td>178</td>
      <td>200</td>
      <td>212</td>
      <td>226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Nepal</td>
      <td>28.1667</td>
      <td>84.2500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Malaysia</td>
      <td>2.5000</td>
      <td>112.5000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>...</td>
      <td>83</td>
      <td>93</td>
      <td>99</td>
      <td>117</td>
      <td>129</td>
      <td>149</td>
      <td>149</td>
      <td>197</td>
      <td>238</td>
      <td>428</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
deaths_reported.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>3/6/20</th>
      <th>3/7/20</th>
      <th>3/8/20</th>
      <th>3/9/20</th>
      <th>3/10/20</th>
      <th>3/11/20</th>
      <th>3/12/20</th>
      <th>3/13/20</th>
      <th>3/14/20</th>
      <th>3/15/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Thailand</td>
      <td>15.0000</td>
      <td>101.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Japan</td>
      <td>36.0000</td>
      <td>138.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>6</td>
      <td>6</td>
      <td>6</td>
      <td>10</td>
      <td>10</td>
      <td>15</td>
      <td>16</td>
      <td>19</td>
      <td>22</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Singapore</td>
      <td>1.2833</td>
      <td>103.8333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Nepal</td>
      <td>28.1667</td>
      <td>84.2500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Malaysia</td>
      <td>2.5000</td>
      <td>112.5000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
recovered_cases.head()
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>Long</th>
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>...</th>
      <th>3/6/20</th>
      <th>3/7/20</th>
      <th>3/8/20</th>
      <th>3/9/20</th>
      <th>3/10/20</th>
      <th>3/11/20</th>
      <th>3/12/20</th>
      <th>3/13/20</th>
      <th>3/14/20</th>
      <th>3/15/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Thailand</td>
      <td>15.0000</td>
      <td>101.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>31</td>
      <td>33</td>
      <td>34</td>
      <td>34</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Japan</td>
      <td>36.0000</td>
      <td>138.0000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>46</td>
      <td>76</td>
      <td>76</td>
      <td>76</td>
      <td>101</td>
      <td>118</td>
      <td>118</td>
      <td>118</td>
      <td>118</td>
      <td>118</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>Singapore</td>
      <td>1.2833</td>
      <td>103.8333</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>96</td>
      <td>96</td>
      <td>97</td>
      <td>105</td>
      <td>105</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>Nepal</td>
      <td>28.1667</td>
      <td>84.2500</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>Malaysia</td>
      <td>2.5000</td>
      <td>112.5000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>23</td>
      <td>24</td>
      <td>24</td>
      <td>24</td>
      <td>26</td>
      <td>26</td>
      <td>26</td>
      <td>35</td>
      <td>42</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
#Extracting all columns
cols = confirmed_cases.keys()
cols
```




    Index(['Province/State', 'Country/Region', 'Lat', 'Long', '1/22/20', '1/23/20',
           '1/24/20', '1/25/20', '1/26/20', '1/27/20', '1/28/20', '1/29/20',
           '1/30/20', '1/31/20', '2/1/20', '2/2/20', '2/3/20', '2/4/20', '2/5/20',
           '2/6/20', '2/7/20', '2/8/20', '2/9/20', '2/10/20', '2/11/20', '2/12/20',
           '2/13/20', '2/14/20', '2/15/20', '2/16/20', '2/17/20', '2/18/20',
           '2/19/20', '2/20/20', '2/21/20', '2/22/20', '2/23/20', '2/24/20',
           '2/25/20', '2/26/20', '2/27/20', '2/28/20', '2/29/20', '3/1/20',
           '3/2/20', '3/3/20', '3/4/20', '3/5/20', '3/6/20', '3/7/20', '3/8/20',
           '3/9/20', '3/10/20', '3/11/20', '3/12/20', '3/13/20', '3/14/20',
           '3/15/20'],
          dtype='object')




```python
#Extracting only the dates columns that have info of confirmed, deaths and recovered cases
confirmed = confirmed_cases.loc[:, cols[4]:cols[-1]]
```


```python
deaths = deaths_reported.loc[:, cols[4]:cols[-1]]
```


```python
recoveries = recovered_cases.loc[:, cols[4]:cols[-1]]
```


```python
#Check the outbreak cases
confirmed.head()
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>1/24/20</th>
      <th>1/25/20</th>
      <th>1/26/20</th>
      <th>1/27/20</th>
      <th>1/28/20</th>
      <th>1/29/20</th>
      <th>1/30/20</th>
      <th>1/31/20</th>
      <th>...</th>
      <th>3/6/20</th>
      <th>3/7/20</th>
      <th>3/8/20</th>
      <th>3/9/20</th>
      <th>3/10/20</th>
      <th>3/11/20</th>
      <th>3/12/20</th>
      <th>3/13/20</th>
      <th>3/14/20</th>
      <th>3/15/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>7</td>
      <td>8</td>
      <td>8</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>19</td>
      <td>...</td>
      <td>48</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>53</td>
      <td>59</td>
      <td>70</td>
      <td>75</td>
      <td>82</td>
      <td>114</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>7</td>
      <td>11</td>
      <td>15</td>
      <td>...</td>
      <td>420</td>
      <td>461</td>
      <td>502</td>
      <td>511</td>
      <td>581</td>
      <td>639</td>
      <td>639</td>
      <td>701</td>
      <td>773</td>
      <td>839</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>7</td>
      <td>7</td>
      <td>10</td>
      <td>13</td>
      <td>...</td>
      <td>130</td>
      <td>138</td>
      <td>150</td>
      <td>150</td>
      <td>160</td>
      <td>178</td>
      <td>178</td>
      <td>200</td>
      <td>212</td>
      <td>226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>7</td>
      <td>8</td>
      <td>8</td>
      <td>...</td>
      <td>83</td>
      <td>93</td>
      <td>99</td>
      <td>117</td>
      <td>129</td>
      <td>149</td>
      <td>149</td>
      <td>197</td>
      <td>238</td>
      <td>428</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 54 columns</p>
</div>




```python
#Finding the total confirmed cases, death cases and the recovered casees and append them to an 4 emplty lists
#Also calculate the total mortality rate which is the death_sum/confirmed cases
dates = confirmed.keys()
world_cases   = []
total_deaths  = []
mortality_rate =[]
total_recovered = []

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)


```


```python
confirmed_sum
```




    167449




```python
death_sum
```




    6440




```python
recovered_sum
```




    76034




```python
world_cases
```




    [555,
     653,
     941,
     1434,
     2118,
     2927,
     5578,
     6166,
     8234,
     9927,
     12038,
     16787,
     19881,
     23892,
     27635,
     30817,
     34391,
     37120,
     40150,
     42762,
     44802,
     45221,
     60368,
     66885,
     69030,
     71224,
     73258,
     75136,
     75639,
     76197,
     76823,
     78579,
     78965,
     79568,
     80413,
     81395,
     82754,
     84120,
     86011,
     88369,
     90306,
     92840,
     95120,
     97882,
     101784,
     105821,
     109795,
     113561,
     118592,
     125865,
     128343,
     145193,
     156097,
     167449]




```python
days_since_1_22 = np.array([i for i in range (len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1,1)
total_deathsc = np.array(total_deaths).reshape(-1,1)
total_recovered = np.array(total_recovered).reshape(-1,1)
```


```python
days_since_1_22
```




    array([[ 0],
           [ 1],
           [ 2],
           [ 3],
           [ 4],
           [ 5],
           [ 6],
           [ 7],
           [ 8],
           [ 9],
           [10],
           [11],
           [12],
           [13],
           [14],
           [15],
           [16],
           [17],
           [18],
           [19],
           [20],
           [21],
           [22],
           [23],
           [24],
           [25],
           [26],
           [27],
           [28],
           [29],
           [30],
           [31],
           [32],
           [33],
           [34],
           [35],
           [36],
           [37],
           [38],
           [39],
           [40],
           [41],
           [42],
           [43],
           [44],
           [45],
           [46],
           [47],
           [48],
           [49],
           [50],
           [51],
           [52],
           [53]])




```python
world_cases
```




    array([[   555],
           [   653],
           [   941],
           [  1434],
           [  2118],
           [  2927],
           [  5578],
           [  6166],
           [  8234],
           [  9927],
           [ 12038],
           [ 16787],
           [ 19881],
           [ 23892],
           [ 27635],
           [ 30817],
           [ 34391],
           [ 37120],
           [ 40150],
           [ 42762],
           [ 44802],
           [ 45221],
           [ 60368],
           [ 66885],
           [ 69030],
           [ 71224],
           [ 73258],
           [ 75136],
           [ 75639],
           [ 76197],
           [ 76823],
           [ 78579],
           [ 78965],
           [ 79568],
           [ 80413],
           [ 81395],
           [ 82754],
           [ 84120],
           [ 86011],
           [ 88369],
           [ 90306],
           [ 92840],
           [ 95120],
           [ 97882],
           [101784],
           [105821],
           [109795],
           [113561],
           [118592],
           [125865],
           [128343],
           [145193],
           [156097],
           [167449]])




```python
total_deaths
```




    [17,
     18,
     26,
     42,
     56,
     82,
     131,
     133,
     171,
     213,
     259,
     362,
     426,
     492,
     564,
     634,
     719,
     806,
     906,
     1013,
     1113,
     1118,
     1371,
     1523,
     1666,
     1770,
     1868,
     2007,
     2122,
     2247,
     2251,
     2458,
     2469,
     2629,
     2708,
     2770,
     2814,
     2872,
     2941,
     2996,
     3085,
     3160,
     3254,
     3348,
     3460,
     3558,
     3802,
     3988,
     4262,
     4615,
     4720,
     5404,
     5819,
     6440]




```python
total_recovered
```




    array([[   28],
           [   30],
           [   36],
           [   39],
           [   52],
           [   61],
           [  107],
           [  126],
           [  143],
           [  222],
           [  284],
           [  472],
           [  623],
           [  852],
           [ 1124],
           [ 1487],
           [ 2011],
           [ 2616],
           [ 3244],
           [ 3946],
           [ 4683],
           [ 5150],
           [ 6295],
           [ 8058],
           [ 9395],
           [10865],
           [12583],
           [14352],
           [16121],
           [18177],
           [18890],
           [22886],
           [23394],
           [25227],
           [27905],
           [30384],
           [33277],
           [36711],
           [39782],
           [42716],
           [45602],
           [48228],
           [51170],
           [53796],
           [55865],
           [58358],
           [60694],
           [62494],
           [64404],
           [67003],
           [68324],
           [70251],
           [72624],
           [76034]])




```python
#Future forcasting for the next 10 days
days_in_future = 10
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast[:-10]
```


```python
future_forecast
```




    array([[ 0],
           [ 1],
           [ 2],
           [ 3],
           [ 4],
           [ 5],
           [ 6],
           [ 7],
           [ 8],
           [ 9],
           [10],
           [11],
           [12],
           [13],
           [14],
           [15],
           [16],
           [17],
           [18],
           [19],
           [20],
           [21],
           [22],
           [23],
           [24],
           [25],
           [26],
           [27],
           [28],
           [29],
           [30],
           [31],
           [32],
           [33],
           [34],
           [35],
           [36],
           [37],
           [38],
           [39],
           [40],
           [41],
           [42],
           [43],
           [44],
           [45],
           [46],
           [47],
           [48],
           [49],
           [50],
           [51],
           [52],
           [53],
           [54],
           [55],
           [56],
           [57],
           [58],
           [59],
           [60],
           [61],
           [62],
           [63]])




```python
#Converting all integers into datetime for better visulaization
start = '1/22/2020'
start_date = datetime.datetime.strptime(start,'%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/sv/jpl0h46174j92wdy3xnymc5m0000gn/T/ipykernel_2523/1157045006.py in <module>
          3 start_date = datetime.datetime.strptime(start,'%m/%d/%Y')
          4 future_forcast_dates = []
    ----> 5 for i in range(len(future_forcast)):
          6     future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


    NameError: name 'future_forcast' is not defined



```python
#For visualizing with the latest data of 15th of march
latest_confirmed = confirmed_cases[dates[-1]]
latest_deaths = deaths_reported[dates[-1]]
latest_recoveries = recovered_cases[dates[-1]]
```


```python
latest_confirmed  #This gives all the values from 15th of march
```




    0      114
    1      839
    2      226
    3        1
    4      428
          ... 
    445      1
    446      1
    447      1
    448      1
    449      1
    Name: 3/15/20, Length: 450, dtype: int64




```python
latest_deaths #Gives total deaths that were reported from 15th of march across all the regions
```




    0       1
    1      22
    2       0
    3       0
    4       0
           ..
    445     0
    446     0
    447     0
    448     0
    449     0
    Name: 3/15/20, Length: 450, dtype: int64




```python
latest_recoveries
```




    0       35
    1      118
    2      105
    3        1
    4       42
          ... 
    445      0
    446      0
    447      0
    448      0
    449      0
    Name: 3/15/20, Length: 450, dtype: int64




```python
#Figure out the list of unique countries using .uniques
unique_countries = list(confirmed_cases['Country/Region'].unique())

```


```python
unique_countries
```




    ['Thailand',
     'Japan',
     'Singapore',
     'Nepal',
     'Malaysia',
     'Canada',
     'Australia',
     'Cambodia',
     'Sri Lanka',
     'Germany',
     'Finland',
     'United Arab Emirates',
     'Philippines',
     'India',
     'Italy',
     'Sweden',
     'Spain',
     'Belgium',
     'Egypt',
     'Lebanon',
     'Iraq',
     'Oman',
     'Afghanistan',
     'Bahrain',
     'Kuwait',
     'Algeria',
     'Croatia',
     'Switzerland',
     'Austria',
     'Israel',
     'Pakistan',
     'Brazil',
     'Georgia',
     'Greece',
     'North Macedonia',
     'Norway',
     'Romania',
     'Estonia',
     'Netherlands',
     'San Marino',
     'Belarus',
     'Iceland',
     'Lithuania',
     'Mexico',
     'New Zealand',
     'Nigeria',
     'Ireland',
     'Luxembourg',
     'Monaco',
     'Qatar',
     'Ecuador',
     'Azerbaijan',
     'Armenia',
     'Dominican Republic',
     'Indonesia',
     'Portugal',
     'Andorra',
     'Latvia',
     'Morocco',
     'Saudi Arabia',
     'Senegal',
     'Argentina',
     'Chile',
     'Jordan',
     'Ukraine',
     'Hungary',
     'Liechtenstein',
     'Poland',
     'Tunisia',
     'Bosnia and Herzegovina',
     'Slovenia',
     'South Africa',
     'Bhutan',
     'Cameroon',
     'Colombia',
     'Costa Rica',
     'Peru',
     'Serbia',
     'Slovakia',
     'Togo',
     'Malta',
     'Martinique',
     'Bulgaria',
     'Maldives',
     'Bangladesh',
     'Paraguay',
     'Albania',
     'Cyprus',
     'Brunei',
     'US',
     'Burkina Faso',
     'Holy See',
     'Mongolia',
     'Panama',
     'China',
     'Iran',
     'Korea, South',
     'France',
     'Cruise Ship',
     'Denmark',
     'Czechia',
     'Taiwan*',
     'Vietnam',
     'Russia',
     'Moldova',
     'Bolivia',
     'Honduras',
     'United Kingdom',
     'Congo (Kinshasa)',
     "Cote d'Ivoire",
     'Jamaica',
     'Reunion',
     'Turkey',
     'Cuba',
     'Guyana',
     'Kazakhstan',
     'Cayman Islands',
     'Guadeloupe',
     'Ethiopia',
     'Sudan',
     'Guinea',
     'Aruba',
     'Kenya',
     'Antigua and Barbuda',
     'Uruguay',
     'Ghana',
     'Jersey',
     'Namibia',
     'Seychelles',
     'Trinidad and Tobago',
     'Venezuela',
     'Curacao',
     'Eswatini',
     'Gabon',
     'Guatemala',
     'Guernsey',
     'Mauritania',
     'Rwanda',
     'Saint Lucia',
     'Saint Vincent and the Grenadines',
     'Suriname',
     'occupied Palestinian territory',
     'Kosovo',
     'Central African Republic',
     'Congo (Brazzaville)',
     'Equatorial Guinea',
     'Uzbekistan']




```python
#Calculating total number of confirmed cases by each country
country_confirmed_cases = []
no_cases = []
for i in unique_countries:
    cases = latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases > 0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
        
for i in no_cases:
    unique_countries.remove(i)
    
unique_countries = [k for k,v in sorted(zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1),reverse = True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()

```


```python
#number of cases per country/region
print('Confirmed Cases by Countries/Regions:')
for i in range(len(unique_countries)):
   print(f'{unique_countries[i]}: {country_confirmed_cases[i]} cases')
```

    Confirmed Cases by Countries/Regions:
    China: 81003 cases
    Italy: 24747 cases
    Iran: 13938 cases
    Korea, South: 8162 cases
    Spain: 7798 cases
    Germany: 5795 cases
    France: 4513 cases
    US: 3499 cases
    Switzerland: 2200 cases
    Norway: 1221 cases
    United Kingdom: 1144 cases
    Netherlands: 1135 cases
    Sweden: 1022 cases
    Belgium: 886 cases
    Denmark: 875 cases
    Austria: 860 cases
    Japan: 839 cases
    Cruise Ship: 696 cases
    Malaysia: 428 cases
    Qatar: 401 cases
    Greece: 331 cases
    Australia: 297 cases
    Czechia: 253 cases
    Canada: 252 cases
    Israel: 251 cases
    Portugal: 245 cases
    Finland: 244 cases
    Singapore: 226 cases
    Slovenia: 219 cases
    Bahrain: 214 cases
    Estonia: 171 cases
    Iceland: 171 cases
    Brazil: 162 cases
    Philippines: 140 cases
    Romania: 131 cases
    Ireland: 129 cases
    Poland: 119 cases
    Indonesia: 117 cases
    Iraq: 116 cases
    Thailand: 114 cases
    India: 113 cases
    Kuwait: 112 cases
    Egypt: 110 cases
    Lebanon: 110 cases
    Saudi Arabia: 103 cases
    San Marino: 101 cases
    United Arab Emirates: 98 cases
    Chile: 74 cases
    Russia: 63 cases
    Luxembourg: 59 cases
    Taiwan*: 59 cases
    Vietnam: 56 cases
    Slovakia: 54 cases
    Pakistan: 53 cases
    South Africa: 51 cases
    Bulgaria: 51 cases
    Brunei: 50 cases
    Croatia: 49 cases
    Algeria: 48 cases
    Serbia: 48 cases
    Argentina: 45 cases
    Peru: 43 cases
    Panama: 43 cases
    Albania: 42 cases
    Mexico: 41 cases
    Colombia: 34 cases
    Georgia: 33 cases
    Hungary: 32 cases
    Latvia: 30 cases
    Ecuador: 28 cases
    Morocco: 28 cases
    Belarus: 27 cases
    Costa Rica: 27 cases
    Armenia: 26 cases
    Cyprus: 26 cases
    Senegal: 24 cases
    Bosnia and Herzegovina: 24 cases
    Azerbaijan: 23 cases
    Moldova: 23 cases
    Oman: 22 cases
    Malta: 21 cases
    Sri Lanka: 18 cases
    Tunisia: 18 cases
    Afghanistan: 16 cases
    North Macedonia: 14 cases
    Maldives: 13 cases
    Lithuania: 12 cases
    Dominican Republic: 11 cases
    Bolivia: 10 cases
    Jamaica: 10 cases
    Venezuela: 10 cases
    Martinique: 9 cases
    Kazakhstan: 9 cases
    New Zealand: 8 cases
    Jordan: 8 cases
    Cambodia: 7 cases
    Reunion: 7 cases
    Paraguay: 6 cases
    Turkey: 6 cases
    Ghana: 6 cases
    Bangladesh: 5 cases
    Liechtenstein: 4 cases
    Cuba: 4 cases
    Guyana: 4 cases
    Uruguay: 4 cases
    Ukraine: 3 cases
    Burkina Faso: 3 cases
    Honduras: 3 cases
    Guadeloupe: 3 cases
    Kenya: 3 cases
    Nigeria: 2 cases
    Monaco: 2 cases
    Cameroon: 2 cases
    Congo (Kinshasa): 2 cases
    Aruba: 2 cases
    Jersey: 2 cases
    Namibia: 2 cases
    Seychelles: 2 cases
    Trinidad and Tobago: 2 cases
    Saint Lucia: 2 cases
    Kosovo: 2 cases
    Nepal: 1 cases
    Andorra: 1 cases
    Bhutan: 1 cases
    Togo: 1 cases
    Holy See: 1 cases
    Mongolia: 1 cases
    Cote d'Ivoire: 1 cases
    Cayman Islands: 1 cases
    Ethiopia: 1 cases
    Sudan: 1 cases
    Guinea: 1 cases
    Antigua and Barbuda: 1 cases
    Curacao: 1 cases
    Eswatini: 1 cases
    Gabon: 1 cases
    Guatemala: 1 cases
    Guernsey: 1 cases
    Mauritania: 1 cases
    Rwanda: 1 cases
    Saint Vincent and the Grenadines: 1 cases
    Suriname: 1 cases
    Central African Republic: 1 cases
    Congo (Brazzaville): 1 cases
    Equatorial Guinea: 1 cases
    Uzbekistan: 1 cases



```python
#Finding the list of unique provinces
unique_provinces = list(confirmed_cases['Province/State'].unique())
#UK, Denmark are france are considered provinces which we are removing
outliers = ['United Kingdom', 'Denmark', 'France']
for i in outliers:
    unique_provinces.remove(i)
```


```python
#Finding the number of confirmed case from province, city or state
province_confirmed_cases = []
no_cases = []
for i in unique_provinces:
    cases = latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if cases > 0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
            
for i in no_cases:     
        unique_provinces.remove(i)
```


```python
#Finding the number of cases per province
for i in range(len(unique_provinces)):
    print(f'{unique_provinces[i]}: {province_confirmed_cases[i]} cases')
```

    British Columbia: 73 cases
    New South Wales: 134 cases
    Victoria: 57 cases
    Queensland: 61 cases
    South Australia: 20 cases
    Western Australia: 17 cases
    Tasmania: 6 cases
    Northern Territory: 1 cases
    Ontario: 104 cases
    Alberta: 39 cases
    Quebec: 24 cases
    Washington: 643 cases
    New York: 732 cases
    California: 426 cases
    Massachusetts: 164 cases
    Diamond Princess: 742 cases
    Grand Princess: 23 cases
    Georgia: 99 cases
    Colorado: 131 cases
    Florida: 115 cases
    New Jersey: 98 cases
    Oregon: 36 cases
    Texas: 72 cases
    Illinois: 93 cases
    Pennsylvania: 66 cases
    Iowa: 18 cases
    Maryland: 32 cases
    North Carolina: 33 cases
    South Carolina: 28 cases
    Tennessee: 39 cases
    Virginia: 45 cases
    Arizona: 13 cases
    Indiana: 20 cases
    Kentucky: 20 cases
    District of Columbia: 16 cases
    Nevada: 24 cases
    New Hampshire: 13 cases
    Minnesota: 35 cases
    Nebraska: 17 cases
    Ohio: 37 cases
    Rhode Island: 20 cases
    Wisconsin: 32 cases
    Connecticut: 24 cases
    Hawaii: 6 cases
    Oklahoma: 7 cases
    Utah: 28 cases
    Kansas: 8 cases
    Louisiana: 91 cases
    Missouri: 5 cases
    Vermont: 8 cases
    Alaska: 1 cases
    Arkansas: 16 cases
    Delaware: 7 cases
    Idaho: 5 cases
    Maine: 12 cases
    Michigan: 33 cases
    Mississippi: 10 cases
    Montana: 7 cases
    New Mexico: 13 cases
    North Dakota: 1 cases
    South Dakota: 9 cases
    Wyoming: 3 cases
    Hubei: 67794 cases
    Guangdong: 1360 cases
    Henan: 1273 cases
    Zhejiang: 1231 cases
    Hunan: 1018 cases
    Anhui: 990 cases
    Jiangxi: 935 cases
    Shandong: 760 cases
    Jiangsu: 631 cases
    Chongqing: 576 cases
    Sichuan: 539 cases
    Heilongjiang: 482 cases
    Beijing: 442 cases
    Shanghai: 353 cases
    Hebei: 318 cases
    Fujian: 296 cases
    Guangxi: 252 cases
    Shaanxi: 245 cases
    Yunnan: 174 cases
    Hainan: 168 cases
    Guizhou: 146 cases
    Tianjin: 136 cases
    Shanxi: 133 cases
    Gansu: 133 cases
    Hong Kong: 145 cases
    Liaoning: 125 cases
    Jilin: 93 cases
    Xinjiang: 76 cases
    Inner Mongolia: 75 cases
    Ningxia: 75 cases
    Qinghai: 18 cases
    Macau: 10 cases
    Faroe Islands: 11 cases
    St Martin: 2 cases
    Channel Islands: 3 cases
    New Brunswick: 2 cases
    Tibet: 1 cases
    Saint Barthelemy: 1 cases
    Gibraltar: 1 cases
    Australian Capital Territory: 1 cases
    French Polynesia: 3 cases
    Manitoba: 4 cases
    Saskatchewan: 2 cases
    Alabama: 12 cases
    Puerto Rico: 5 cases
    Virgin Islands, U.S.: 1 cases
    French Guiana: 7 cases
    Guam: 3 cases
    Newfoundland and Labrador: 1 cases
    Prince Edward Island: 1 cases
    Mayotte: 1 cases



```python
#handling non values if there is any
nan_indices = []

for i in range(len(unique_provinces)):
    if type(unique_provinces[i]) == float:
        nan_indices.append(i)
        
unique_provinces = list(unique_provinces)
province_confirmed_cases = list(province_confirmed_cases)

for i in nan_indices:
    unique_provinces.pop(i)
    province_confirmed_cases.pop(i)
```


```python
#Plotting a bar graph
plt.figure(figsize=(32, 32))
plt.barh(unique_countries, country_confirmed_cases)
plt.title('Number of Covid-19 Confirmed Cases in Countries')
plt.xlabel('Number of Covid19 Confirmed Cases')
plt.show()
```


    
![png](output_38_0.png)
    



```python
#Plot a graph to see the total confirmed cases between mainland china and outside mai land china
china_confirmed = latest_confirmed[confirmed_cases['Country/Region']=='China'].sum()
outside_mainland_china_confirmed = np.sum(country_confirmed_cases)- china_confirmed
plt.figure(figsize =(16, 9))
plt.barh('Mainland China', china_confirmed)
plt.barh('Outside Mainland China', outside_mainland_china_confirmed)
plt.title('Number of Confirmed CoronaViruse Cases')
plt.show()

```


    
![png](output_39_0.png)
    



```python
print ('Outside Mainland China {} cases :'.format(outside_mainland_china_confirmed))
print('Mainland China: {} cases'.format(china_confirmed))
print('Total: {} cases'.format(china_confirmed+outside_mainland_china_confirmed))
```

    Outside Mainland China 86446 cases :
    Mainland China: 81003 cases
    Total: 167449 cases



```python
#Only show 10 countries with the most confirmed cases
visual_unique_countries = []
visual_confirmed_cases = []
others = np.sum(country_confirmed_cases[10:])
for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
    
visual_unique_countries.append('Others')
visual_confirmed_cases.append(others)
```


```python
#Visualize the 10 countries
plt.figure(figsize = (32, 18))
plt.barh(visual_unique_countries, visual_confirmed_cases)
plt.title('Number of Covid-19 confirmed cases in Countries/Regions', size = 20)
plt.show()
```


    
![png](output_42_0.png)
    



```python
#Creating pie chart
c = random.choices(list(mcolors.CSS4_COLORS.values()), k = len(unique_countries))
plt.figure(figsize=(20,20))
plt.title('Covid-19 Confirmed Cases per Country')
plt.pie(visual_confirmed_cases, colors=c)
plt.legend(visual_unique_countries, loc='best')
plt.show()
```


    
![png](output_43_0.png)
    



```python
#Creating pie chart to visualize total 10 countries apart from China
c = random.choices(list(mcolors.CSS4_COLORS.values()), k = len(unique_countries))
plt.figure(figsize=(20,20))
plt.title('Covid-19 Confirmed Cases per Country')
plt.pie(visual_confirmed_cases[1:], colors=c)
plt.legend(visual_unique_countries[1:], loc='best')
plt.show()
```


    
![png](output_44_0.png)
    



```python
import seaborn as sns
```


```python
sns.pairplot(confirmed_cases,hue = 'test')
```


```python
import seaborn as sns
sns.pairplot(df,hue = 'test')
```


```python

```


```python

```
