# Mutual Information Neural Estimation (MINE) [(paper)](https://arxiv.org/abs/1801.04062)
Implement MINE with TensorFlow 2

## Toy Example
Calculate mutual information between X and Y, denoted by **I(X ; Y)**.  
Let X be sampled from Normal distribution, and Y be random variable denoting the result of the rolling a dice.  
In theory, they are independent so **I(X ; Y) = 0**, the experimental result is shown below.


## Results
* Runs in 10000 iterations in our experiment.
* Loss value is decreasing, the updating process is reasonable.
* The lower bound gradually flattens around 0, meeting the theoretical result.
<p align="center"><img src="https://github.com/joe-ip-ml/MINE/blob/main/img/loss.png" width="80%"></p>
<p align="center"><img src="https://github.com/joe-ip-ml/MINE/blob/main/img/lb.png" width="80%"></p> 
