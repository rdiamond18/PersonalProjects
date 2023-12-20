# Intro
In January 2023, I applied for a summer internship with the NY Mets. After submitting a resume and cover letter, I was asked to complete a take home assignment. While I did not get the job after submitting the take home, I wanted to share my work on GitHub. I was asked two questions and provided two data sets, and I will provide details about the questions, my process and my solutions in this notebook.

# Q1 - Predicting pitches
In this question, I was given various information about a large data set of pitches and the resulting pitch type of those pitches. I was given the inning, boolean variables such as is_bottom, is_lhp and is_lhb (which have true values of 1 and a false values of zero) and situational variables such as count, outs, score and basecode (which indicates how many runners were on base and where they were located). With this info, I was asked to develop a model to predict the pitch type of a testing set. Below is my solution.


```python
#load packages
import numpy as np
import pandas as pd

#import training and testing sets
train = pd.read_csv('/Users/richarddiamond/Desktop/mets/materials/Q1_Pitches_train.csv')
test = pd.read_csv('/Users/richarddiamond/Desktop/mets/materials/Q1_Pitches_test.csv')

train.head()
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
      <th>inning</th>
      <th>is_bottom</th>
      <th>balls</th>
      <th>strikes</th>
      <th>outs_before</th>
      <th>is_lhp</th>
      <th>is_lhb</th>
      <th>pitch_type</th>
      <th>bat_score_before</th>
      <th>field_score</th>
      <th>basecode_before</th>
      <th>batterid</th>
      <th>pitcherid</th>
      <th>cid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>FF</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>347</td>
      <td>1304</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>FF</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>269</td>
      <td>1661</td>
      <td>2052</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>FT</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>43</td>
      <td>1048</td>
      <td>2029</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>FF</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>98</td>
      <td>1521</td>
      <td>2049</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>SL</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>460</td>
      <td>1100</td>
      <td>2050</td>
    </tr>
  </tbody>
</table>
</div>



## Identify Features and Target Variable



```python
features = ['inning', 'is_bottom', 'balls', 'strikes', 'outs_before', 'is_lhp', 'is_lhb',
            'bat_score_before', 'field_score', 'basecode_before', 'batterid', 'pitcherid', 'cid']

#separate train df into features and target variables
X_train = train.loc[:, features].values
y_train = train.loc[:,['pitch_type']].values

#sseparate features out of test df
X_test = test.loc[:, features].values
```

## Standardize Data
Since the scales of the feature variables vary, we must standardize the data using StandardScaler.


```python
from sklearn.preprocessing import StandardScaler 

#scale and transform data 
sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test) 
```

## Use K-Nearest Neighbors to separate data into classes
This technique will use distance in a higher dimensional space to give probabilities that each pitch belongs to the Fourseam Fastball, Twoseam Fastball, Changeup, Curveball and Slider Classes


```python
from sklearn.neighbors import KNeighborsClassifier

#using 5 neighbors as we have 5 pitch types
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train) 

#classifying all pitches will take about 5 mins
y_predict = classifier.predict_proba(X_test)

#example of the prediction probabilities of one pitch
print(y_predict[0])

#print this to know we're finished 
print('done!')
```

    /opt/anaconda3/lib/python3.9/site-packages/sklearn/neighbors/_classification.py:198: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return self._fit(X, y)


    [0.2 0.  0.6 0.2 0. ]
    done!


## Assign predictions to new DataFrame
Finally, let's input our predictions into the new DataFrame.


```python
#create a data frame out of our predictions
pitchDF = pd.DataFrame(y_predict, columns = ['FF','FT','CB', 'SL', 'CH'])
```


```python
#assign values of pitchDF to columns in test
test['FF'] = pitchDF['FF']
test['FT'] = pitchDF['FT']
test['CB'] = pitchDF['CB']
test['SL'] = pitchDF['SL']
test['CH'] = pitchDF['CH']

#take a look at the new test DF
test.head()
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
      <th>inning</th>
      <th>is_bottom</th>
      <th>balls</th>
      <th>strikes</th>
      <th>outs_before</th>
      <th>is_lhp</th>
      <th>is_lhb</th>
      <th>bat_score_before</th>
      <th>field_score</th>
      <th>basecode_before</th>
      <th>batterid</th>
      <th>pitcherid</th>
      <th>cid</th>
      <th>FF</th>
      <th>FT</th>
      <th>CB</th>
      <th>SL</th>
      <th>CH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1000</td>
      <td>2000</td>
      <td>0.2</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1001</td>
      <td>2001</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1001</td>
      <td>2001</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1001</td>
      <td>2001</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>1001</td>
      <td>2001</td>
      <td>0.0</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.2</td>
      <td>0.4</td>
    </tr>
  </tbody>
</table>
</div>



### Question to answer: What other information would you have wanted to improve your predictions’ performance?

I would have wanted more information about the pitchers themselves. If a pitcher throws one certain pitch more frequently than others, that information could be very useful in these predictions as each pitcher has specific tendencies independent of game situation. If I had this information, I would have factored it into my K-Nearest Neihbors calculation as one of the feature variables that the algorithm uses to make its predeictions.

# Q2 - Hot Dog Sales
In this question, I was asked to rank the effectiveness of 30 hot dog vendors. I was given the number of hot dogs sold, as well as auxillary information such as what section the vendors were in and what day of the week the sales came on.


```python
#import data
salesData = pd.read_csv('/Users/richarddiamond/Desktop/mets/materials/Q2_citi_vendors.csv')
salesData.head()
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
      <th>game</th>
      <th>day</th>
      <th>section</th>
      <th>vendor</th>
      <th>hot_dogs_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>145</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>10</td>
      <td>121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>21</td>
      <td>56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>123</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



## Check if other factors influence hot dog sales
In order to rank the salespeople based on effectiveness, I wanted to take a look at whether day of the week and section influence sales.


```python
#group by day to see if day affects hot dogs sold
daysDF = salesData.groupby(by = 'day').mean()

#we're only looking at hot dogs sold, so we can drop the other columns
daysDF = daysDF.drop(['game', 'section', 'vendor'], axis = 1)

daysDF.head()
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
      <th>hot_dogs_sold</th>
    </tr>
    <tr>
      <th>day</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>112.690909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>125.379167</td>
    </tr>
    <tr>
      <th>3</th>
      <td>120.283333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>127.150000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>130.650000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#group by section to see if day affects hot dogs sold
sectionDF = salesData.groupby(by = 'section').mean()

#we're only looking at hot dogs sold, so we can drop the other columns
sectionDF = sectionDF.drop(['game', 'day', 'vendor'], axis = 1)

sectionDF.head()
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
      <th>hot_dogs_sold</th>
    </tr>
    <tr>
      <th>section</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>151.740741</td>
    </tr>
    <tr>
      <th>2</th>
      <td>127.222222</td>
    </tr>
    <tr>
      <th>3</th>
      <td>127.037037</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.148148</td>
    </tr>
    <tr>
      <th>5</th>
      <td>130.185185</td>
    </tr>
  </tbody>
</table>
</div>



## Developing xHotDogsSold
Since there's enough variance in average sales each day of the week and in each section, I felt it was best to factor those into my calculation. In order to accurately figure out how effective salespeople are, we can take an average of the hot dogs sold on that day of the week and in that section. One caveat is that this metric relies on the assumption that day of the week and section have equal importance, which may not necessarily be true.


```python
#create two dictionaries that give us average values

#for the average of a certain section
sectionDict = sectionDF.to_dict()

#for the average of a certain day
daysDict = daysDF.to_dict()
```


```python
#get values of the day and section given in the DataFrame
dayList = salesData.day.values.tolist()
sectionList = salesData.section.values.tolist()
```


```python
#declare a list that will gold the average of the expected values when combining the day and section
averageList = []

#populate list using values from the lists that were created
for x in range(len(dayList)):
    averageList.append((daysDict['hot_dogs_sold'][dayList[x]] + sectionDict['hot_dogs_sold'][sectionList[x]]) / 2)
    
#set values in DataFrame
salesData['x_hot_dogs_sold'] = averageList

#show Dataframe
salesData.head()
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
      <th>game</th>
      <th>day</th>
      <th>section</th>
      <th>vendor</th>
      <th>hot_dogs_sold</th>
      <th>x_hot_dogs_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>6</td>
      <td>145</td>
      <td>138.559954</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>10</td>
      <td>121</td>
      <td>126.300694</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
      <td>21</td>
      <td>56</td>
      <td>126.208102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>123</td>
      <td>122.763657</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>100</td>
      <td>127.782176</td>
    </tr>
  </tbody>
</table>
</div>




```python
#calculate difference between expected and actual sales
salesData['sold_over_expected'] = salesData['x_hot_dogs_sold'] - salesData['hot_dogs_sold']

#show DataFrame
salesData.head()
```


```python
#get averages of sold_over_expected for each vendor and sort by value
vendorDF = salesData.groupby(by = 'vendor').mean()
vendorDF = vendorDF.sort_values(by = 'sold_over_expected', ascending = False)

vendorDF
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
      <th>game</th>
      <th>day</th>
      <th>section</th>
      <th>hot_dogs_sold</th>
      <th>x_hot_dogs_sold</th>
      <th>sold_over_expected</th>
    </tr>
    <tr>
      <th>vendor</th>
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
      <th>26</th>
      <td>35.333333</td>
      <td>4.055556</td>
      <td>13.722222</td>
      <td>104.944444</td>
      <td>136.043711</td>
      <td>31.099267</td>
    </tr>
    <tr>
      <th>12</th>
      <td>41.229730</td>
      <td>4.108108</td>
      <td>10.851351</td>
      <td>107.000000</td>
      <td>132.574191</td>
      <td>25.574191</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41.103896</td>
      <td>4.012987</td>
      <td>9.259740</td>
      <td>108.779221</td>
      <td>132.429944</td>
      <td>23.650723</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41.052632</td>
      <td>3.921053</td>
      <td>10.342105</td>
      <td>115.526316</td>
      <td>131.834277</td>
      <td>16.307961</td>
    </tr>
    <tr>
      <th>13</th>
      <td>41.527778</td>
      <td>3.833333</td>
      <td>9.555556</td>
      <td>116.097222</td>
      <td>131.128812</td>
      <td>15.031590</td>
    </tr>
    <tr>
      <th>18</th>
      <td>42.527027</td>
      <td>3.986486</td>
      <td>9.648649</td>
      <td>119.905405</td>
      <td>132.433040</td>
      <td>12.527635</td>
    </tr>
    <tr>
      <th>25</th>
      <td>35.600000</td>
      <td>3.933333</td>
      <td>13.133333</td>
      <td>121.066667</td>
      <td>132.619809</td>
      <td>11.553143</td>
    </tr>
    <tr>
      <th>23</th>
      <td>47.086957</td>
      <td>3.956522</td>
      <td>14.913043</td>
      <td>120.130435</td>
      <td>131.422796</td>
      <td>11.292361</td>
    </tr>
    <tr>
      <th>7</th>
      <td>40.786667</td>
      <td>3.893333</td>
      <td>10.466667</td>
      <td>121.706667</td>
      <td>132.618005</td>
      <td>10.911339</td>
    </tr>
    <tr>
      <th>8</th>
      <td>42.014286</td>
      <td>4.014286</td>
      <td>8.828571</td>
      <td>121.728571</td>
      <td>131.193847</td>
      <td>9.465276</td>
    </tr>
    <tr>
      <th>19</th>
      <td>41.342857</td>
      <td>4.042857</td>
      <td>9.485714</td>
      <td>126.800000</td>
      <td>133.692488</td>
      <td>6.892488</td>
    </tr>
    <tr>
      <th>22</th>
      <td>33.785714</td>
      <td>4.035714</td>
      <td>11.964286</td>
      <td>127.500000</td>
      <td>131.903753</td>
      <td>4.403753</td>
    </tr>
    <tr>
      <th>17</th>
      <td>43.242857</td>
      <td>3.942857</td>
      <td>11.442857</td>
      <td>128.742857</td>
      <td>132.277718</td>
      <td>3.534861</td>
    </tr>
    <tr>
      <th>11</th>
      <td>41.067568</td>
      <td>3.945946</td>
      <td>10.000000</td>
      <td>129.554054</td>
      <td>132.732341</td>
      <td>3.178287</td>
    </tr>
    <tr>
      <th>10</th>
      <td>41.657534</td>
      <td>4.109589</td>
      <td>10.054795</td>
      <td>131.506849</td>
      <td>132.527280</td>
      <td>1.020430</td>
    </tr>
    <tr>
      <th>9</th>
      <td>41.240000</td>
      <td>3.973333</td>
      <td>9.880000</td>
      <td>131.800000</td>
      <td>132.498477</td>
      <td>0.698477</td>
    </tr>
    <tr>
      <th>30</th>
      <td>39.000000</td>
      <td>2.666667</td>
      <td>15.500000</td>
      <td>129.500000</td>
      <td>128.695835</td>
      <td>-0.804165</td>
    </tr>
    <tr>
      <th>14</th>
      <td>41.855072</td>
      <td>4.000000</td>
      <td>10.231884</td>
      <td>135.376812</td>
      <td>132.267142</td>
      <td>-3.109670</td>
    </tr>
    <tr>
      <th>28</th>
      <td>30.500000</td>
      <td>2.800000</td>
      <td>11.500000</td>
      <td>134.200000</td>
      <td>128.386615</td>
      <td>-5.813385</td>
    </tr>
    <tr>
      <th>15</th>
      <td>42.014085</td>
      <td>3.873239</td>
      <td>10.788732</td>
      <td>139.154930</td>
      <td>132.054247</td>
      <td>-7.100682</td>
    </tr>
    <tr>
      <th>2</th>
      <td>42.686567</td>
      <td>3.985075</td>
      <td>10.089552</td>
      <td>140.820896</td>
      <td>132.584272</td>
      <td>-8.236624</td>
    </tr>
    <tr>
      <th>5</th>
      <td>41.191781</td>
      <td>4.027397</td>
      <td>10.712329</td>
      <td>142.178082</td>
      <td>131.990617</td>
      <td>-10.187465</td>
    </tr>
    <tr>
      <th>16</th>
      <td>40.783784</td>
      <td>3.945946</td>
      <td>9.581081</td>
      <td>145.675676</td>
      <td>132.205050</td>
      <td>-13.470626</td>
    </tr>
    <tr>
      <th>27</th>
      <td>42.052632</td>
      <td>3.631579</td>
      <td>11.789474</td>
      <td>147.473684</td>
      <td>129.066147</td>
      <td>-18.407537</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40.295775</td>
      <td>4.028169</td>
      <td>10.239437</td>
      <td>151.957746</td>
      <td>131.445767</td>
      <td>-20.511979</td>
    </tr>
    <tr>
      <th>24</th>
      <td>39.476190</td>
      <td>4.142857</td>
      <td>15.666667</td>
      <td>155.238095</td>
      <td>133.899474</td>
      <td>-21.338621</td>
    </tr>
    <tr>
      <th>6</th>
      <td>39.600000</td>
      <td>4.106667</td>
      <td>11.346667</td>
      <td>156.906667</td>
      <td>133.595971</td>
      <td>-23.310696</td>
    </tr>
    <tr>
      <th>21</th>
      <td>35.666667</td>
      <td>4.000000</td>
      <td>13.916667</td>
      <td>159.750000</td>
      <td>134.774810</td>
      <td>-24.975190</td>
    </tr>
    <tr>
      <th>20</th>
      <td>40.430556</td>
      <td>4.000000</td>
      <td>10.250000</td>
      <td>170.986111</td>
      <td>133.885101</td>
      <td>-37.101010</td>
    </tr>
    <tr>
      <th>29</th>
      <td>33.250000</td>
      <td>4.500000</td>
      <td>12.750000</td>
      <td>208.500000</td>
      <td>132.471393</td>
      <td>-76.028607</td>
    </tr>
  </tbody>
</table>
</div>



### Final Results
Based on the data, the 5 best salespeople are vendors 26, 12, 4, 1 and 13 in that order. The 5 worst, starting at the least effective, are 29, 20, 21, 6 and 24.
