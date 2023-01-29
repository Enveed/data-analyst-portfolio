# Ethereum Prediction with Linear Regression

## Overview

__[Ethereum](https://ethereum.org)__ is a decentralized, open-source blockchain with smart contract functionality. Ether is the native cryptocurrency of the platform. Among cryptocurrencies, ether is second only to bitcoin in market capitalization. Ethereum was conceived in 2013 by programmer Vitalik Buterin. <br> <br>
In this project, we will attempt to predict the closing price of **Ethereum** by using a Linear Regression model, which is being trained on 80% of the dataset of **Ethereum** historical prices from January 2018 to December 2022. We will use **"Open" price** and **"Date"** as the independent variables in order to predict the **"Close" price**.

## Code

### Importing the libraries

In this section, we will import all the essential libraries that are needed for this project. The most notable ones should be: **numpy**, **pandas**, **seaborn** and **matplotlib**.


```python
# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date

%matplotlib inline
```

### Importing the dataset

We will be using Ethereum's historical price data from __[Yahoo Finance](https://finance.yahoo.com/quote/ETH-USD/history?p=ETH-USD)__ from January 1st, 2018 to December 31st, 2022. The file is also included within this project folder in order to avoid any mismatch.


```python
# Reading the data and parsing the 'Date' field
eth_price = pd.read_csv("ETH-USD.csv", usecols=["Date", "Open", "Close"], parse_dates=["Date"])
eth_price = eth_price.sort_values("Date")
eth_price.head()
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
      <th>Date</th>
      <th>Open</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01</td>
      <td>755.757019</td>
      <td>772.640991</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-02</td>
      <td>772.346008</td>
      <td>884.443970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-03</td>
      <td>886.000000</td>
      <td>962.719971</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-04</td>
      <td>961.713013</td>
      <td>980.921997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-05</td>
      <td>975.750000</td>
      <td>997.719971</td>
    </tr>
  </tbody>
</table>
</div>



### Exporatory Data Analysis (EDA)

Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.


```python
# Getting the number of rows and columns
eth_price.shape
```




    (1826, 3)




```python
# Showing the statistics of numeric variables
eth_price.describe()
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
      <th>Open</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1826.000000</td>
      <td>1826.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1147.241063</td>
      <td>1147.253520</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1200.784975</td>
      <td>1200.263833</td>
    </tr>
    <tr>
      <th>min</th>
      <td>84.279694</td>
      <td>84.308296</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>209.032078</td>
      <td>208.920326</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>519.065552</td>
      <td>518.846069</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1802.362610</td>
      <td>1803.337372</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4810.071289</td>
      <td>4812.087402</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Exploring the data type of each column
eth_price.dtypes
```




    Date     datetime64[ns]
    Open            float64
    Close           float64
    dtype: object




```python
# Checking for empty columns
eth_price.isna().sum()
```




    Date     0
    Open     0
    Close    0
    dtype: int64




```python
# Checking for correlation between different quantitative columns
sns.heatmap(eth_price.corr(), annot=True)
```




    <AxesSubplot:>




    
![png](output_16_1.png)
    



```python
# Visualize the data
plt.figure(figsize = (20,12))
sns.lineplot(x="Date", y="value", hue="variable", data=pd.melt(eth_price, ["Date"])).set(title="Ethereum price over time")
plt.ylabel("Price")
```




    Text(0, 0.5, 'Price')




    
![png](output_17_1.png)
    


### Modeling the dataset

Since linear regression model does not recognize timestamp, we will have to convert the timestamp into ordinal values.


```python
eth_price['Date'] = pd.to_datetime(eth_price['Date']).apply(lambda date: date.toordinal())
eth_price
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
      <th>Date</th>
      <th>Open</th>
      <th>Close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>736695</td>
      <td>755.757019</td>
      <td>772.640991</td>
    </tr>
    <tr>
      <th>1</th>
      <td>736696</td>
      <td>772.346008</td>
      <td>884.443970</td>
    </tr>
    <tr>
      <th>2</th>
      <td>736697</td>
      <td>886.000000</td>
      <td>962.719971</td>
    </tr>
    <tr>
      <th>3</th>
      <td>736698</td>
      <td>961.713013</td>
      <td>980.921997</td>
    </tr>
    <tr>
      <th>4</th>
      <td>736699</td>
      <td>975.750000</td>
      <td>997.719971</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1821</th>
      <td>738516</td>
      <td>1226.987061</td>
      <td>1212.791626</td>
    </tr>
    <tr>
      <th>1822</th>
      <td>738517</td>
      <td>1212.736572</td>
      <td>1189.986084</td>
    </tr>
    <tr>
      <th>1823</th>
      <td>738518</td>
      <td>1190.010132</td>
      <td>1201.595337</td>
    </tr>
    <tr>
      <th>1824</th>
      <td>738519</td>
      <td>1201.569580</td>
      <td>1199.232788</td>
    </tr>
    <tr>
      <th>1825</th>
      <td>738520</td>
      <td>1199.360107</td>
      <td>1196.771240</td>
    </tr>
  </tbody>
</table>
<p>1826 rows × 3 columns</p>
</div>




```python
import matplotlib.ticker as mticker
eda_plot = eth_price.plot(x='Date', y='Close', figsize=(20,6))
eda_plot.set_title("Ethereum price over time")
eda_plot.set_xlabel("Date")
eda_plot.set_ylabel("Close Price (USD)")
new_labels = [date.fromordinal(int(item)) for item in eda_plot.get_xticks()]
eda_plot.set_xticks(eda_plot.get_xticks())
eda_plot.set_xticklabels(new_labels)
```




    [Text(736500.0, 0, '2017-06-20'),
     Text(736750.0, 0, '2018-02-25'),
     Text(737000.0, 0, '2018-11-02'),
     Text(737250.0, 0, '2019-07-10'),
     Text(737500.0, 0, '2020-03-16'),
     Text(737750.0, 0, '2020-11-21'),
     Text(738000.0, 0, '2021-07-29'),
     Text(738250.0, 0, '2022-04-05'),
     Text(738500.0, 0, '2022-12-11'),
     Text(738750.0, 0, '2023-08-18')]




    
![png](output_21_1.png)
    



```python
# Splitting the dataset to prepare for training
x = eth_price.loc[:, ["Date", "Open"]].values
y = eth_price.iloc[:, 2].values
```

### Training the dataset


```python
# Splitting the dataset into the training set (80%) and test set (20%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
```


```python
# Fitting the training datasets into the Linear Regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
```




    LinearRegression()




```python
# Using the trained LR model to predict the dependent variables (y) based on the x values
y_pred = regressor.predict(x_test)
y_pred
```




    array([ 227.15853491,  229.39940328,  681.62675625,  182.40180466,
            247.80451904,  240.37180729,   87.61989374, 1300.48061478,
            159.86421508, 1038.8773955 ,  168.31188375, 1878.6995361 ,
           3010.64479212,  433.49017446,  245.58934946, 3280.81937647,
            141.7307689 ,  452.68574356,  377.54635259, 2552.80642518,
            175.68579122,  202.66920749,  187.27351986,  291.45220711,
            185.61013067, 1012.45691085,  182.23185487,  324.18943907,
            588.64165123, 4624.78133561, 1224.9614218 , 1290.7817175 ,
            219.48513272, 3676.11059815,  224.24825431,  167.23778038,
           2271.75906809, 2541.16301664, 3324.81559827, 1139.37467695,
           4152.6799789 , 1715.36520462,  454.75688151, 1690.31973003,
           1573.99003035,  368.69264032,  272.94748417, 2812.34286725,
            237.5458369 , 1024.79616159, 1459.62888553, 1716.51542247,
           1877.40090555, 2554.24788952,  192.42754663,  127.1632084 ,
            224.99344551,  227.21862094, 3375.74907445,  501.63438115,
            523.35876557,  186.12961026, 4127.67019981,   92.98148477,
           1279.35896839, 1619.05657345, 3158.47135036, 2727.12214672,
           2169.02560288,  192.45915133,  220.60869058,  371.87031815,
           2351.72909419,  560.08497407, 1278.26675337,  231.75419538,
           2940.17525709,  295.87757527, 1531.91862153,  357.46861382,
            146.65455773, 3168.58980343,  483.22670772, 1701.07010677,
            222.93889101,  102.05530321, 2615.74118915,  781.95678434,
            638.59955467,  356.36786848,  608.42678015, 2660.31238934,
            270.28576802,  183.2769201 ,  417.38541539, 1216.57162492,
            549.64143063,  179.68478712,  137.5261631 ,  163.45014251,
           2432.59404396,  407.0577755 ,  960.95427979,  163.551662  ,
            705.7934599 , 2894.0973523 ,  181.76897398,  514.62023972,
            560.67789862,  180.74178358, 1317.39989369, 2824.38856986,
           4075.69488098,  316.37784497, 3091.02118775,  281.70153879,
            286.84922642,  472.86816002,  186.94543222,  941.16985454,
           1677.08411198,  238.65959277,  320.36668154,  808.52386556,
           1658.94900388,  256.11923485, 1614.4033974 ,  404.26076733,
            270.00717463,  225.59357549, 1359.93299535,  205.56515761,
           2945.77510467, 1766.59230839,  180.20359523, 3605.8877974 ,
           1790.45726388,  145.37952388,  167.08067451,  377.89085769,
           1522.17335385, 3786.94335577,  107.85537166, 1538.35727114,
           1172.01368988,  242.7367115 ,  787.36876407,  138.3127597 ,
            227.69808152, 3391.03763191, 4029.28196912,  165.23460718,
           1303.10063695,  578.6843118 , 1230.76509658,  177.2935962 ,
           3073.63379413, 1620.90518428, 1916.34879291, 3869.7347957 ,
           1555.20720422, 2505.20824477, 2590.68022872,  168.91262526,
            697.94186549,  162.67191947,  221.66010754,  308.80869753,
           1816.07341444,  836.30501511, 3379.8663052 , 2998.61021597,
            235.80515318,  138.0663552 , 3822.07109912, 1366.46625653,
           1814.76900372, 2805.12233097, 1313.70809597,  269.55526698,
            203.69019937,  265.13879384,  409.48305406,  231.10732838,
            385.0467298 ,  193.82857553,  229.53662586,  815.32729937,
           1578.43355339, 3101.98188171,  183.25981322,  471.2383577 ,
            749.78687976, 3137.19948028, 1967.4065914 , 3857.69568715,
           2618.40612202,  371.08848879,  354.45581419,  117.84290767,
           2921.7240956 ,  991.19223717,  874.65307331,  135.84626466,
           1236.11988908,  512.53080064, 3580.84351278,  177.7180436 ,
            146.11360136, 2140.31909332,  198.02305974, 1937.30040626,
            557.49896971,  409.7530045 ,  174.61840871, 3052.95612434,
            124.34498397,  385.99205538, 2211.67857586, 3366.36611201,
            306.89485012, 3779.59920026,  186.53259309,  174.99091105,
            385.97260518,  600.27780701, 1324.06133592, 2364.2608822 ,
           2436.19531012,  227.23953743, 2115.26287777, 4093.01566577,
           1153.74058593, 1040.87171387, 3234.99091633, 4262.12426039,
            173.61438201,  858.35310599, 1942.21637653, 2633.62737455,
           2530.13274989, 1128.63448006, 1825.02650876, 1921.99591752,
           1295.27736495,  264.91614184,  487.86165504,  184.45383597,
            641.23777838,  148.15051864,  211.93825003, 1235.77257733,
            212.73021438,  540.63509074,  210.42173097,  118.53730632,
           1281.06445144, 1885.40926876,  147.45555576, 1331.20647899,
           4560.06780569,  161.55419777,  684.3148642 ,  340.89660014,
            219.22820351,  385.48009804, 2923.42125041, 3228.44507243,
           1320.94287203, 2996.25972348, 3260.21126387, 2023.48607354,
            846.05789524,  187.32199751,  599.11830843,  212.1107503 ,
           2767.68194739,  576.14616653,  198.51037119, 1811.18004481,
            363.12802604,  474.97960778,  277.58089414, 3597.93527684,
            177.65927857,  134.81494131, 1222.31066291, 1609.22454289,
           3099.6389914 ,  398.61614559,  157.63875085,  198.31260215,
           1806.3258791 ,  382.28242159,  392.26274032,  212.99795629,
            409.26105197,  147.48599549,  394.56680436,  368.51960204,
            221.8394221 ,  344.84268485, 1514.21833532,  393.58754978,
           3243.36752196, 1255.30185255,  971.85471059,  450.11165618,
           1866.80541723,  379.20002792, 3006.84606141,  208.41640435,
            438.7151288 ,  167.2149187 ,  257.40802968, 1116.92487655,
           1190.06747455,  266.65008954, 1693.17889249,  218.77845209,
           1818.05719115, 1115.07546572,  145.14810498,  394.47738093,
            219.12299408, 1263.41235099, 3816.99798599,  432.30023683,
           1812.51983418, 2644.84519682,  146.9024134 ,  225.46771477,
            272.87450293,  178.55764776,  380.10977988,  412.78782347,
            393.64162724, 1254.80111117,  739.85209391, 3176.43001459,
           1715.45708936,  431.16125448,  175.03077559, 3901.80036455,
            159.27740908,  578.34364773,  165.39257886, 2108.88462608,
           1375.14279159,  150.6512359 ,  186.178288  ,  117.45734153,
            390.75714871,  219.50091642, 3878.09047054, 1733.90733527,
            165.89387061,  197.57396676,  106.01868894,  355.44968932,
            534.95357028,  128.50474431,  129.11299182, 1493.44298624,
            229.14377294,  282.44333008])




```python
# Calculating the Coefficients
print(regressor.coef_)
```

    [0.00473041 0.99633087]
    


```python
# Calculating the Intercept
print(regressor.intercept_)
```

    -3485.200264532437
    

### Evaluating the model

In order to evaluate the model, we will be using a statistical measure called R-Squared method that shows how well the data fit the regression model (the goodness of fit).


```python
#Calculating the R squared value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```




    0.9936097930328858



We can see that the model was able to predict the **"Close" price** with an accuracy of about 99.36%.

## Conclusion

Overall, this project does not express the ability to build an intensive machine learning model that can blindly predict the crypto prices. It rather serves as a project that can show the application of Linear Regression model by the author.
