# MATH165 Final Project
### By Richard Diamond
This notebook will host my code and walk you through my final project for MATH165.

In section 9 of the textbook, problem 7 asks the following:

You are given the opportunity to bid on a mystery box containing a mystery prize! The value of the prize is completely unknown, except that it is worth at least nothing and at most a million dollars. So the true value V of the prize is considered to be the uniform on [0,1] (measured in millions of dollars).  
You can choose to bid any nonnegative amount b (in millions of dollars). If $b < \frac{1/4}{V}$, then your bid is rejected and nothing is gained or lost. If $b < \frac{1/4}{V}$, then your bid is accepted and your net payoff is $V - b$.  
Find the optimal bid $b$ to maximize expected payoff.

## 1. Simulating the original problem via monte-carlo
I wanted to use a monte-carlo simulation, where we simulate our experiment a large number of times generating random numbers, to prove our result from the homework, which was that 0.25 (or $250,000) is the most ideal bid. To do this, I wanted to generate a long list of random prize values in python, and then see how different bid amounts would fare with these randomly generated values.


```python
#import packages
import numpy as np
import pandas as pd
```


```python
#this function that will generate list of random values according to the uniform distribution
def generate_prizes(amount):
    return np.random.uniform(0,1,amount)
    
```


```python
#generate list of bid amounts to check against our random prizes
bid_amounts = np.linspace(0,1,101)
```

Now, we will create a dataframe in pandas to hold our resulting payoffs for each bid.


```python
bid_results = pd.DataFrame(columns = ['bid_amount', 'average_payoff'])

bid_results['bid_amount'] = bid_amounts

#show empty dataframe
bid_results.head()
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



### Compute average payoff given bid amount
Now, we will write a function that will take in a bid amount and compute our average payoff.


```python
def get_payoff(bid_amount, prize_list):
    #variable to hold cumulative winnings
    cumulative_winnings = 0
    
    #iterate through prize list and compute outcome
    for prize in prize_list:
        
        #payoff is 0 if our bid is less than 1/4 of the value of the prize
        if bid_amount < prize * (1/4):
            cumulative_winnings += 0
        #otherwise, get payoff from subtracting bid from value
        else:
            result = prize - bid_amount
            cumulative_winnings += result
    
    #get average result by dividing by number of trials
    return cumulative_winnings / len(prize_list)
```

### Get payoff values with different bid amounts
We now will iterate through our list of bid amounts and get our resulting average prize amounts.


```python
#get list of 100000 prizes to ensure a large enough sample
prizes = generate_prizes(100000)

#create empty list to hold result payoffs 
payoffs = []

#iterate through our list of bid amounts and run them through our function
for bid in bid_amounts:
    average_payoff = get_payoff(bid, prizes)
    payoffs.append(average_payoff)
```


```python
#add average payoffs to dataframe
bid_results['average_payoff'] = payoffs
```


```python
#sort values
bid_results.sort_values(by = 'average_payoff', ascending = False)
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0.25</td>
      <td>0.249203</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.26</td>
      <td>0.239203</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.24</td>
      <td>0.229528</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.27</td>
      <td>0.229203</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.28</td>
      <td>0.219203</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.96</td>
      <td>-0.460797</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.97</td>
      <td>-0.470797</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.98</td>
      <td>-0.480797</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.99</td>
      <td>-0.490797</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>-0.500797</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 2 columns</p>
</div>



As we see in the above table, the best bid is 0.25, just like we proved in the homework. I then wanted to check on the distribution of the average payoffs, and see if we could observe any interesting patterns.


```python
#import plotting package
import matplotlib.pyplot as plt
```


```python
#Create visualization
plt.figure(figsize=(10,6))
plt.plot(bid_results['bid_amount'], bid_results['average_payoff'])
plt.title("Bid Amount vs. Average Payoff - Uniform")
plt.xlabel("Bid Amount (millions of dollars)")
plt.ylabel('Payoff(millions of dollars)')
plt.axvline(0.25, color = 'red')
```




    <matplotlib.lines.Line2D at 0x7f89a9a1f8b0>




    
![png](output_16_1.png)
    


### Discussion
As expected, we got the same result as the homework: 0.25 was the best bid, and generated about 250,000 dollars on average. Also as expected, the way the function decreased differently for values less than and greater than 0.25. On the less than side, we see an exponential decrease, and on the greater than side we see a linear decrease. This is due to fact that for any value greater than 0.25, we will never have a bid greater than $\frac{1}{4}$ of the prize, so we'd expect a linear decrease where all that matters is subtracting the bid from the prize.

## 2. Using the normal distribution to generate a prize
I then wanted to use the normal distribution to generate a prize instead of a uniform distribution. This would mean that we would have both values greater than 1 million dollars and perhaps negative values. However, I decided to keep the range between 0 and 1, so I filtered out results outside of that range.  
To summarize, in this alteration of our problem, we are generating a random prize according to the normal distribution as opposed to the uniform distribution, restricting our prizes to the interval [0,1].


```python
#write new generate function
def generate_prizes_normal(amount):
    
    #generate normal values with mean 0.5 and standard deviation 0.2
    prizes = np.random.normal(0.5,0.2,amount)
    prizes_filtered = filter(lambda x: x>= 0 and x <= 1, prizes)
    
    return list(prizes_filtered)
```


```python
#create new dataframe to hold results for normal distribution
bid_results_normal = pd.DataFrame(columns = ['bid_amount', 'average_payoff'])

bid_results_normal['bid_amount'] = bid_amounts

#show empty dataframe
bid_results_normal.head()
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get list of 100000 normally distributed prizes to ensure a large enough sample
normal_prizes = generate_prizes_normal(100000)

#create empty list to hold result payoffs 
normal_payoffs = []

#iterate through our list of bid amounts and run them through our function
for bid in bid_amounts:
    average_payoff = get_payoff(bid, normal_prizes)
    normal_payoffs.append(average_payoff)
```


```python
#add average payoffs to dataframe
bid_results_normal['average_payoff'] = normal_payoffs
```


```python
#sort values
bid_results_normal.sort_values(by = 'average_payoff', ascending = False)
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>0.22</td>
      <td>0.264384</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.21</td>
      <td>0.263801</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.23</td>
      <td>0.261667</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.20</td>
      <td>0.259723</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.24</td>
      <td>0.256826</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.96</td>
      <td>-0.459647</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.97</td>
      <td>-0.469647</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.98</td>
      <td>-0.479647</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.99</td>
      <td>-0.489647</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>-0.499647</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 2 columns</p>
</div>



Our best bid has now fallen to 220,000 dollars, but our best average payoffs have increased, indicating that this is a slightly better game to play than the original version. Next, I wanted to plot the results from each bid amount, just like what was done above.


```python
#Create visualization
plt.figure(figsize=(10,6))
plt.plot(bid_results_normal['bid_amount'], bid_results_normal['average_payoff'])
plt.title("Bid Amount vs. Average Payoff")
plt.xlabel("Bid Amount (millions of dollars)")
plt.ylabel('Payoff(millions of dollars)')
plt.axvline(0.22, color = 'red')
```




    <matplotlib.lines.Line2D at 0x7f89582d1fa0>




    
![png](output_25_1.png)
    


The graph looks very similar but does not have as sharp of a corner around the maximum. 

## 3. Using the exponential distribution
I then wanted to try to capture the same essence of the problem, but this time use the exponential distribution. Because of the nature of the exponential, I will leave the prizes without an upper bound, theoretically being able to increase way past the original 1 million dollar cap.
To summarize, in this alteration of our problem, we are generating a random prize according to the exponential distribution as opposed to the uniform distribution.


```python
def generate_prizes_exponential(amount):
    
    #generate exponential with mean 0.5
    return np.random.exponential(0.5, amount)
```

Now, we will follow our same process as above and test all of our bid amounts from 0 to 1.


```python
#create new dataframe to hold results for normal distribution
bid_results_exp = pd.DataFrame(columns = ['bid_amount', 'average_payoff'])

bid_results_exp['bid_amount'] = bid_amounts

#show empty dataframe
bid_results_exp.head()
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get list of 100000 exponentially distributed prizes to ensure a large enough sample
exp_prizes = generate_prizes_exponential(100000)

#create empty list to hold result payoffs 
exp_payoffs = []

#iterate through our list of bid amounts and run them through our function
for bid in bid_amounts:
    average_payoff = get_payoff(bid, exp_prizes)
    exp_payoffs.append(average_payoff)
```


```python
#add average payoffs to dataframe
bid_results_exp['average_payoff'] = exp_payoffs
```


```python
#sort values
bid_results_exp.sort_values(by = 'average_payoff', ascending = False)
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>0.24</td>
      <td>0.081480</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.25</td>
      <td>0.081275</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.23</td>
      <td>0.080809</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.26</td>
      <td>0.080336</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.22</td>
      <td>0.080034</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.96</td>
      <td>-0.461646</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.97</td>
      <td>-0.471584</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.98</td>
      <td>-0.481492</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.99</td>
      <td>-0.491399</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>-0.501306</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 2 columns</p>
</div>




```python
#Create visualization
plt.figure(figsize=(10,6))
plt.plot(bid_results_exp['bid_amount'], bid_results_exp['average_payoff'])
plt.title("Bid Amount vs. Average Payoff - Exponential")
plt.xlabel("Bid Amount (millions of dollars)")
plt.ylabel('Payoff(millions of dollars)')
plt.axvline(0.23, color = 'red')
```




    <matplotlib.lines.Line2D at 0x7f89788940d0>




    
![png](output_34_1.png)
    


## 4. Comparing all distributions
Next, I wanted to create a chart that would overlay the results of all three games on top of each other to visualize how they differ in their payouts.


```python
#plot data
plt.figure(figsize=(10,6))
plt.plot(bid_amounts, bid_results['average_payoff'], label = 'Uniform')
plt.plot(bid_amounts, bid_results_exp['average_payoff'], label = 'Exponential')
plt.plot(bid_amounts, bid_results_normal['average_payoff'], label = 'Normal', color = 'red')
plt.title('Average payoffs by distribution')
plt.xlabel('Bid amount')
plt.ylabel('Average Payoff')
plt.legend() 
plt.show()
```


    
![png](output_36_0.png)
    


### Discussion
Overall, the normal distribution game is the best one to play among the three. They all seem to descend similarly after peaking in the low 0.20s, but they have different and interesting patterns when descending into values less than 0.2. The average payoffs with the normally distributed prizes seem to descend in a way that looks kind of like the normal distibution after peaking at 0.22, which was very interesting to me. When looking at the exponentially and uniformly distributed prizes, we see a more exponential decrease as we approach zero from the right hand side of their peaks.  
Overall, the strategy seems clear among all distributions: the bidder should stay below 0.25. Interestingly, this value is the same 1/4 value that is the threshold for our bid getting accepted: you will only get your bid accpeted if it is greater than a quarter of the value. I wanted to try the same problem with a different threshold to see if we would find a similar relationship between optimal bid and threshold.

## 5. Changing the threshold to 1/3
I wanted to go back to the uniform distribution problem, but now make the bidder have to bid at least half the value of the prize to see their bid accepted. This should lead to lower payouts across the board, and what I'm interested in is whether the optimal bid amount will increase to about $\frac{1}{3}$.

I first needed to write a new function to get our payoffs that takes in a variable threshold instead of a fixed threshold of $\frac{1}{4}$. The function is the exact same but now takes in an extra parameter and uses the variable threshold.


```python
def get_payoff_variable(bid_amount, prize_list, threshold):
    #variable to hold cumulative winnings
    cumulative_winnings = 0
    
    #iterate through prize list and compute outcome
    for prize in prize_list:
        
        #payoff is 0 if our bid is less than 1/4 of the value of the prize
        if bid_amount < prize * (threshold):
            cumulative_winnings += 0
        #otherwise, get payoff from subtracting bid from value
        else:
            result = prize - bid_amount
            cumulative_winnings += result
    
    #get average result by dividing by number of trials
    return cumulative_winnings / len(prize_list)
```

Now, I wanted to repeat the same process as the uniform distributon process above.


```python
bid_results_variable_threshold = pd.DataFrame(columns = ['bid_amount', 'average_payoff'])

bid_results_variable_threshold['bid_amount'] = bid_amounts

#show empty dataframe
bid_results_variable_threshold.head()
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
#get list of 100000 prizes to ensure a large enough sample
prizes = generate_prizes(100000)

#create empty list to hold result payoffs 
payoffs_variable = []

#iterate through our list of bid amounts and run them through our function
for bid in bid_amounts:
    #pass in 1/2 as our threshold for our bid being accepted
    average_payoff = get_payoff_variable(bid, prizes, 1/3)
    payoffs_variable.append(average_payoff)
```


```python
#add average payoffs to dataframe
bid_results_variable_threshold['average_payoff'] = payoffs_variable
```


```python
#sort values
bid_results_variable_threshold.sort_values(by = 'average_payoff', ascending = False)
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
      <th>bid_amount</th>
      <th>average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>33</th>
      <td>0.33</td>
      <td>0.163766</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.34</td>
      <td>0.160349</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.32</td>
      <td>0.153844</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.35</td>
      <td>0.150349</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.31</td>
      <td>0.144023</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.96</td>
      <td>-0.459651</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.97</td>
      <td>-0.469651</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.98</td>
      <td>-0.479651</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.99</td>
      <td>-0.489651</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>-0.499651</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 2 columns</p>
</div>



Interestingly, the ideal bid is now around $\frac{1}{3}$, which is the same as our bid acceptance threshold. This further supports the idea that the optimal strategy in this game is to make a big enough bid so that your bid won't be rejected, but the smallest possible bid that satisfies that condition, meaning you want to bid around what the threshold is for a uniformly distributed prize.

### Plotting our results


```python
#Create visualization
plt.figure(figsize=(10,6))
plt.plot(bid_results_variable_threshold['bid_amount'], bid_results_variable_threshold['average_payoff'])
plt.title("Bid Amount vs. Average Payoff - Uniform (Threshold = 1/3)")
plt.xlabel("Bid Amount (millions of dollars)")
plt.ylabel('Payoff(millions of dollars)')
plt.axvline(0.333333, color = 'red')
```




    <matplotlib.lines.Line2D at 0x7f89487e97c0>




    
![png](output_48_1.png)
    


This plot tells the same story as above and shows that we will have a similar distribution centered around our threshold.

## 6. Introducing a second bidder
Lastly, I wanted to simulate a situation in which we had a second bidder and see what would happen if we changed the parameters for our problem slightly.  
In this case, we will make our bidder smart, but not exactly optimal. We will do this by generating a normally distributed value with the mean being our optimal bid of 0.25. We will also return to our original problem, where if you bid less than $\frac{1}{4}$ of the total, your bid is rejected. Let's also say that if your bid is higher than the opponent, you will get the prize in return, but if you bid less, then your bid will not be taken and you will get a net result of 0.


```python
#create new dataframe to account for tracking adversaries winnings
bid_results_with_opponent = pd.DataFrame(columns = ['bid_amount', 'average_payoff', 'adversary_average_payoff'])

bid_results_with_opponent['bid_amount'] = bid_amounts

#show empty dataframe
bid_results_with_opponent.head()
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
      <th>bid_amount</th>
      <th>average_payoff</th>
      <th>adversary_average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.01</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.04</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Now, we will write a new function to generate our payoffs with our adversary bidding as well.


```python
def get_payoff_adversary(bid_amount, prize_list, adversary_average_bid):
    #variable to hold cumulative winnings for us and adversary
    cumulative_winnings, adversary_cumulative_winnings = 0,0
                         
    #iterate through prize list and compute outcome
    for prize in prize_list:
        
        #generate adversary bid with given mean and variance of 0.25 (always positive)
        adversary_bid = abs(np.random.normal(adversary_average_bid, 0.25))
        
        #check if adversary bid less than 1/4 of the amount
        if adversary_bid < prize * (1/4):
            cumulative_winnings += 0
        
        #if we outbid adversary, adversary loses bid
        elif adversary_bid < bid_amount:
            adversary_cumulative_winnings += 0
        
        #else, adversary gets difference between prize and bid
        else: 
            result_adversary = prize - adversary_bid
            adversary_cumulative_winnings += result_adversary
                         
        
        #repeat same checks for our bid
        if bid_amount < prize * (1/4):
            cumulative_winnings += 0
                         
        #if adversary outbids us, we lose our bid
        elif adversary_bid > bid_amount:
            cumulative_winnings += 0
        #else, we get difference between prize and bid
        else: 
            result = prize - bid_amount
            cumulative_winnings += result
    
    #return our average and adversary average winnings
    avg_winnings = cumulative_winnings / len(prize_list)
    adv_avg_winnings = adversary_cumulative_winnings / len(prize_list)
    
    return (avg_winnings, adv_avg_winnings)
```


```python
#get list of 100000 uniformly distributed prizes to ensure a large enough sample
unif_prizes_adversary = generate_prizes(100000)

#create empty lists to hold result payoffs for us and adversary
unif_payoffs = []
unif_payoffs_adversary = []

#iterate through our list of bid amounts and run them through our function
for bid in bid_amounts:
    average_payoff, average_payoff_adversary = get_payoff_adversary(bid, unif_prizes_adversary, 0.25)
    unif_payoffs.append(average_payoff)
    unif_payoffs_adversary.append(average_payoff_adversary)
```


```python
#add average payoffs to dataframe
bid_results_with_opponent['average_payoff'] = unif_payoffs

bid_results_with_opponent['adversary_average_payoff'] = unif_payoffs_adversary
```

Now, let's show a dataframe containing our results:


```python
bid_results_with_opponent.sort_values(by = 'average_payoff', ascending = False)
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
      <th>bid_amount</th>
      <th>average_payoff</th>
      <th>adversary_average_payoff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0.25</td>
      <td>0.120762</td>
      <td>0.029542</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.26</td>
      <td>0.119027</td>
      <td>0.025539</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.27</td>
      <td>0.117073</td>
      <td>0.022131</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.28</td>
      <td>0.116435</td>
      <td>0.017402</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.29</td>
      <td>0.115653</td>
      <td>0.013454</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.96</td>
      <td>-0.457297</td>
      <td>-0.001126</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.97</td>
      <td>-0.467277</td>
      <td>-0.001137</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.98</td>
      <td>-0.477427</td>
      <td>-0.000972</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.99</td>
      <td>-0.487517</td>
      <td>-0.000869</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>-0.497703</td>
      <td>-0.000653</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 3 columns</p>
</div>



### Discussion
The results with an adversary are similar to those without one. Our optimal bis is still around 0.25, and we can't really glean any advantage from trying to outbid the adversary every time. I wanted to plot the expected payoffs with and without an adversary to observe the difference in expectation between the two scenarios.


```python
#plot data
plt.figure(figsize=(10,6))
plt.plot(bid_amounts, bid_results['average_payoff'], label = 'No adversary')
plt.plot(bid_amounts, bid_results_with_opponent['average_payoff'], label = 'With adversary')
plt.title('Average payoffs with and without adversary')
plt.xlabel('Bid amount')
plt.ylabel('Average Payoff')
plt.axvline(0.25, color = 'red')
plt.legend() 
plt.show()
```


    
![png](output_59_0.png)
    


Our functions look similar, but they each descend from the peak (which is about 0.25) in a different manner. Interestingly, there are some bid amounts greater than 0.5 where the game is actually slightly better to play with the adversary than without. Lastly, I wanted to check if our optimal strategy would change if we were against a conservative bidder.

## 7. Conservative bidder

Let's say our opponent was very conservative and bid a normally distributed amount centered at 0.1 every time. My prediction is that our optimal bid will decrease, as all we need to do is outbid the opponent to see the prize money.


```python
#get list of 100000 uniformly distributed prizes to ensure a large enough sample
unif_prizes_adversary = generate_prizes(100000)

#create empty lists to hold result payoffs for us and adversary
unif_payoffs_cons = []
unif_payoffs_adversary_cons = []

#iterate through our list of bid amounts and run them through our function
for bid in bid_amounts:
    average_payoff, average_payoff_adversary = get_payoff_adversary(bid, unif_prizes_adversary, 0.1)
    unif_payoffs_cons.append(average_payoff)
    unif_payoffs_adversary_cons.append(average_payoff_adversary)
```


```python
#add average payoffs to dataframe
bid_results_with_opponent['average_payoff_conservative'] = unif_payoffs_cons

bid_results_with_opponent['adversary_average_payoff_conservative'] = unif_payoffs_adversary_cons
```


```python
bid_results_with_opponent.sort_values(by = 'average_payoff', ascending = False)
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
      <th>bid_amount</th>
      <th>average_payoff</th>
      <th>adversary_average_payoff</th>
      <th>average_payoff_conservative</th>
      <th>adversary_average_payoff_conservative</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25</th>
      <td>0.25</td>
      <td>0.120762</td>
      <td>0.029542</td>
      <td>0.160970</td>
      <td>0.037490</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.26</td>
      <td>0.119027</td>
      <td>0.025539</td>
      <td>0.158946</td>
      <td>0.032777</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.27</td>
      <td>0.117073</td>
      <td>0.022131</td>
      <td>0.155813</td>
      <td>0.029149</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.28</td>
      <td>0.116435</td>
      <td>0.017402</td>
      <td>0.153973</td>
      <td>0.024197</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.29</td>
      <td>0.115653</td>
      <td>0.013454</td>
      <td>0.150589</td>
      <td>0.020938</td>
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
      <th>96</th>
      <td>0.96</td>
      <td>-0.457297</td>
      <td>-0.001126</td>
      <td>-0.459998</td>
      <td>-0.000149</td>
    </tr>
    <tr>
      <th>97</th>
      <td>0.97</td>
      <td>-0.467277</td>
      <td>-0.001137</td>
      <td>-0.470015</td>
      <td>-0.000137</td>
    </tr>
    <tr>
      <th>98</th>
      <td>0.98</td>
      <td>-0.477427</td>
      <td>-0.000972</td>
      <td>-0.480013</td>
      <td>-0.000132</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.99</td>
      <td>-0.487517</td>
      <td>-0.000869</td>
      <td>-0.490071</td>
      <td>-0.000073</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1.00</td>
      <td>-0.497703</td>
      <td>-0.000653</td>
      <td>-0.500077</td>
      <td>-0.000065</td>
    </tr>
  </tbody>
</table>
<p>101 rows × 5 columns</p>
</div>



The optimal strategy did not change. I wanted to then plot the average payoffs with a conservative bidder in the same plot as above to compare the different strategies in experiments with different parameters.


```python
#plot data
plt.figure(figsize=(10,6))
plt.plot(bid_amounts, bid_results['average_payoff'], label = 'No adversary')
plt.plot(bid_amounts, bid_results_with_opponent['average_payoff'], label = 'With adversary')
plt.plot(bid_amounts, bid_results_with_opponent['average_payoff_conservative'], label = 'With conservative adversary')

plt.title('Average payoffs with and without adversary')
plt.xlabel('Bid amount')
plt.ylabel('Average Payoff')
plt.axvline(0.25, color = 'red')
plt.legend() 
plt.show()
```


    
![png](output_67_0.png)
    


As we can see, the descent follows a similar path, only the payoffs are a little greater. It is very interesting to me that no matter the conditions, our game's payoffs converge to a similar looking distribution if the input prizes are normally distributed.  
This, to me, shows how strong the central limit theorem is. Above, we have a distrubution that isn't one of our named distributions, but according to different parameters (whether we have an adversary, where the prizes are centered at), we will always see convergence to a similar shape given we have 100,000 prizes.

## 8. Takeaways
Here are my takeaways from the experiments that I ran:
- Monte Carlo simulations can be a very powerful tool to validate answers we computed mathematically.
- In this scenario, the idea bid is around what the threshold is for your bid being accepted, no matter how the prizes are distributed. For instance, if yor bid needs to be at least one third of the prize to be accepted, you should bid around one third of the maximum prize.
- The central limit theorem comes into effect here several times, and it can be applied to various distributions, even those that we don't quite have a mathematical definition for.
