# BABIP Project
In this project, I will be attempting to create a model that can predict BABIP using a variety of offensive metrics.

## Step 1 - Retrieve Data
I collected various offensive data from the 2022 season for around 150 hitters. This data includes batted ball metrics, contact spray and speed data.


```python
#import libraries
import numpy as np
import pandas as pd

#download data
df = pd.read_csv('/Users/richarddiamond/Desktop/personalProjects/BABIP.csv')
df = df.dropna()
df.head()
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
      <th>Name</th>
      <th>G</th>
      <th>PA</th>
      <th>BABIP</th>
      <th>LD%</th>
      <th>GB%</th>
      <th>FB%</th>
      <th>HT1</th>
      <th>SprintSpeed</th>
      <th>AvgExitVelo</th>
      <th>BarrelPercentage</th>
      <th>LaunchAngle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aaron Judge</td>
      <td>148</td>
      <td>633</td>
      <td>0.332</td>
      <td>0.23</td>
      <td>0.41</td>
      <td>0.36</td>
      <td>4.66</td>
      <td>27.3</td>
      <td>95.9</td>
      <td>26.5</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Abraham Toro</td>
      <td>95</td>
      <td>375</td>
      <td>0.254</td>
      <td>0.15</td>
      <td>0.42</td>
      <td>0.44</td>
      <td>4.31</td>
      <td>28.0</td>
      <td>87.0</td>
      <td>6.8</td>
      <td>16.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adam Duvall</td>
      <td>146</td>
      <td>555</td>
      <td>0.260</td>
      <td>0.17</td>
      <td>0.30</td>
      <td>0.53</td>
      <td>4.46</td>
      <td>27.9</td>
      <td>88.3</td>
      <td>12.8</td>
      <td>23.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adam Frazier</td>
      <td>155</td>
      <td>639</td>
      <td>0.339</td>
      <td>0.29</td>
      <td>0.41</td>
      <td>0.30</td>
      <td>4.40</td>
      <td>26.8</td>
      <td>85.1</td>
      <td>1.5</td>
      <td>13.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Akil Baddoo</td>
      <td>124</td>
      <td>461</td>
      <td>0.335</td>
      <td>0.21</td>
      <td>0.40</td>
      <td>0.39</td>
      <td>4.12</td>
      <td>28.9</td>
      <td>85.0</td>
      <td>3.6</td>
      <td>14.2</td>
    </tr>
  </tbody>
</table>
</div>



## Step 2 - Split into Train and Test
To create a model, I split the data set into a training and testing set, with the idea of training the model on the training set and testing it on the testing set to confirm accuracy.


```python
from sklearn.model_selection import train_test_split
features = ['G', 'PA', 'LD%', 'GB%', 'FB%', 'HT1', 'SprintSpeed', 'AvgExitVelo', 'BarrelPercentage', 'LaunchAngle']
X = df.loc[:, features].values
y = df.loc[:,['BABIP']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)
```

## Step 3 - Standardize Data
Since the categories are on different scales (90 would be a strong average exit velocity, but far from an optimal average launch angle), I standardized the different data categories.


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

#fit and transform train but only transform test
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

## Step 4 - Perform PCA on Dataset
In order to reduce the dimensions I would need for my regression, I performed Principal Component Analysis. I aimed to reduce the problem to a number of dimensions that would explain 90% of the variance, which turned out to be 5.


```python
from sklearn.decomposition import PCA

#shooting for 90% of variance to be explained by components
pca = PCA(n_components = 0.9)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
#this tells us that we have 5 features that can explane 90% of variance
print(explained_variance)
```

    [0.30137788 0.21348804 0.17569396 0.14530348 0.0915073 ]


## Step 5 - Multiple Regression
I finally performed linear regression on my dataset, and then tested my model.


```python
from sklearn import linear_model
from sklearn import metrics

model = linear_model.LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

meanAbsoluteError = metrics.mean_absolute_error(y_test, predictions)
print(meanAbsoluteError)
```

    0.020351467109317116


To evaluate the model, I used mean absolute error. This number represents the average difference between the actual values and the predicted values. For this example, if a player had a BABIP of .340, the model would, on average, predict their BABIP about 20 points off, so either .320 or .360. This isn't a bad score, but it's not ideal, as 20 points in BABIP is a significant difference.

## Step 6 - Kernel PCA
Since I didn't achieve strong accuracy with standard PCA, I wanted to try kernel PCA. Kernel PCA is usually better for data that lies in a higher dimensional space, which makes sense as I am working with 11 variables.


```python
from sklearn.decomposition import KernelPCA

#Create new train and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.25, random_state=16)

#Using 5 components based on PCA results from earlier
Kernel_pca = KernelPCA(n_components = 5, kernel= "rbf")
X_train2 = Kernel_pca.fit_transform(X_train2)
X_test2 = Kernel_pca.transform(X_test2)
```


```python
#Try model one more time with Kernel PCA
model2 = linear_model.LinearRegression()
model2.fit(X_train2, y_train2)

predictions = model2.predict(X_test2)

meanAbsoluteError = metrics.mean_absolute_error(y_test, predictions)
print(meanAbsoluteError)
```

    0.030046935348102027


Unfortunately, Kernel PCA didn't perform  better. This leaves us with a mean absolute error of about .020 . This doesn't prove that BABIP can't be predicted, it just means that it can't be confidently predicted using just the factors that I chose.

## Acknowledgements
Thanks to FanGraphs and Baseball Savant for the data, and thanks to my friend Brendan Tivnan for reviewing and helping me with my work.
