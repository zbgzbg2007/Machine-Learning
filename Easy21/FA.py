import random
from env import * 

'''
This file implements Sarsa(lambda) with a linear function approximation for 
the game Easy21.

The state consists of two numbers: the first number of dealer and the
sum of the player. If one goes bust, he gets 1 as result.
When one of the two numbers is 0, the state is terminal. When the game
is terminal but non of them is 0, the function step will compare the 
two numbers and return a trivial state that contains 0 for the lost part, 
or both 0 if draw.

The action could be h (for hit) and s (for stick): h will give 
another card to player and s will stop drawing for player. 

'''



def chooseAction(state, theta):
  ''' 
  Choose next action by eps-greedy method for fixed eps = 0.05
  - Input: 
    - state: a tuple of two integers as the current state 
    - theta: a list of float numbers as parameters
  - Output:
    - action: the next action
  '''
  # choose an action
  eps = 0.05
  rand = random.uniform(0, 1)
  if rand <= eps:
    # random choice
    rand = random.randint(0, 1)
    if rand == 0:
      action = 'h' 
    else:
      action = 's' 
  else:
    # greedy choice
    a, b = ApproximateQ(state, 'h', theta), ApproximateQ(state, 's', theta)
    if a > b:
      action = 'h' 
    else:
      action = 's' 
  return action 





def StateToFeature(state, action):
  ''' 
  This function transform from state and action to a feature vector.
  - Input:
    - state: a pair of integers
    - action: a string 
  - Output:
    - feature: a binary list
  '''
  d, p = state
  feature = [0 for i in range(36)]
  a, b, c = list(), list(), 0
  for i in range(3):
    if 1+i*3 <= d and d <= 4+i*3:
      a.append(i)

  for i in range(6):
    if 1+i*3 <= p and p <= 3*i+6:
      b.append(i)
 
  if action == 's':
    c = 0 
  else: 
    c = 1 
  for i in a:
    for j in b:
      feature[i*6+j+c*18] = 1 
  return feature

def ApproximateQ(state, action, theta):
  ''' 
  This function linearly approximates a value-action function with given
  parameter theta.
  - Input:
    - state: a pair of integers
    - action: a string
    - theta: a list of float numbers as parameters
  - Output:
    - value: the function value corresponding the given state and action
  '''
  feature = StateToFeature(state, action)
  ans = 0 
  for i in range(len(theta)):
    ans += theta[i] * feature[i]
  return ans 

def TD_FA(lambdas, k): 
  ''' 
  Sarsa(lambda) control with function approximation by fixed step size and 
  fixed eps-greedy improvement.
  - Input:
    - lambdas: a list of lambda values
    - k: total number of iterations
  - Output:
    - qs: a list of value-action functions for distinct lambdas 
      {0, 0.1, 0.2, ..., 1}.
  '''
  qs = list()
  eps, alpha = 0.05, 0.01
  for lam in lambdas:
    theta = [0 for i in range(36)] # parameters for value-action function
    for _ in range(k):
      et = [0 for i in range(36)] # eligibility traces
      d, p = random.randint(1, 10), random.randint(1, 10) 
      nexts = (d, p)
      nexta = chooseAction(nexts, theta)
      while d != 0 and p != 0:
        state, action = nexts, nexta
        r, nexts = step(state, action)
        d, p = nexts
    
        if d != 0 and p != 0:
          nexta = chooseAction(nexts, theta)
          err = r + ApproximateQ(nexts, nexta, theta) - ApproximateQ(state, action, theta)
        else:
          err = r - ApproximateQ(state, action, theta) 
        feature = StateToFeature(state, action)
        for i in range(len(et)):
          et[i] = lam * et[i] + feature[i]
          theta[i] += alpha * et[i] * err 
    qs.append(theta)
  return qs

def MeanSquareError(qs, q): 
  ''' 
  Compute the mean-squared error between each approximate function in qs and funciton q.
  - Input:
    - qs: a list of parameters for value-action function approximation
    - q: a dictionary for the target value-action function
  - Output:
    - errs: a list of mean-squared error for each function in qs and q.
  '''
  errs = list()
  for p in qs: 
    err = 0 
    for s, a in q:
      feature = StateToFeature(s, a)
      err += (ApproximateQ(s, a, p) - q[(s, a)]) ** 2
    errs.append(err)

  return errs 



# read the optimal function
q = dict()
with open('optimal-Q', 'r') as file1:
  for l in file1:
    nums = l.split()
    s = (int(nums[0]), int(nums[1]))
    q[(s, nums[2])] = float(nums[3])


# compute parameters for distinct lambdas
lambdas = [i * 0.1 for i in range(11)]
qs = TD_FA(lambdas, 1000) 
errs = MeanSquareError(qs, q)
with open('lambdas-errors-FA', 'w') as file2:
  for i in range(11):
    print >>file2, i * 0.1, errs[i]

lambdas = [0, 1]
errs = []
for i in range(20):
  qs = TD_FA(lambdas, (i+1) * 1000)
  # each tuple in the list errs contains the episode number and errors for the functions
  errs.append(((i+1)*1000, MeanSquareError(qs, q)))

with open('learning-curve-FA', 'w') as file3:
  for num, err in errs:
    print >>file3, num, err[0], err[1]
                   
