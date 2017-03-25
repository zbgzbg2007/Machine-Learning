import random
from env import * 

'''
This file implements Monte-Carlo control and Sarsa(lambda) for 
the small game Easy21. The game environment is in the env.py file.

The state consists of two numbers: the first number of dealer and the
sum of the player. If one goes bust, he gets 1 as result.
When one of the two numbers is 0, the state is terminal. When the game
is terminal but non of them is 0, the function step will compare the 
two numbers and return a trivial state that contains 0 for the lost part, 
or both 0 if draw.

The action could be h (for hit) and s (for stick): h will give 
another card to player and s will stop drawing for player. 

'''



def MonteCarlo(k):
  ''' 
  Monte-Carlo control by a time-varying step size and eps-greedy exploration.
  - Input:
    - k: number of total iterations
  - Output:
    - q: a dictionary represent the current value-action function. 
      q[(s, a)] gives the value of function q(s, a).
  '''
  # n0: a constant to decide eps
  n0 = 100 

  # ns: a dictionary from state to numbers. ns[s] gives the times 
  # of the state s is visited
  ns = dict()

  # nsa: a dictionary from (state, action) to numbers. 
  # nsa[(s, a)] gives the times the action a is taken at the state s.
  nsa = dict()

  # value-action function
  q = dict()

  for _i in range(k):
    episode = sample(q, ns, n0) 
    _, __, r = episode[-1]
    for tup in episode:
      state, action, reward = tup 
      q.setdefault((state, action), 0)
      nsa.setdefault((state, action), 0)
      nsa[(state, action)] += 1
      q[(state, action)] += 1./nsa[(state, action)] * (r - q[(state, action)])

  return q

def sample(q, ns, n0):
  ''' 
  Sample one episode from the current policy q by eps-greedy exploration,
  where eps = n0/(n0 + ns[s]) for state s.
  - Input:
    - q: a dictionary represent the current value-action function. 
    - ns: a dictionary from state to numbers. ns[s] gives the times 
      of the state s is visited
    - n0: a constant to decide eps 
  - Output:
    - sample: episode sampled from the current policy. Each episode 
      is a list of tuples of (state, action, reward).
  
  '''
  sample = list()
  d, p = random.randint(1, 10), random.randint(1, 10) 
  r = 0 
  while d != 0 and p != 0:
    state = (d, p)
    ns.setdefault(state, 0)
    ns[state] += 1
    action = chooseAction(q, n0, ns, state)
    # take action
    r, nexts = step(state, action)
    sample.append((state, action, r)) 
    d, p = nexts
    if d == 0 or p == 0:
      return sample

def chooseAction(q, n0, ns, state):
  ''' 
  Choose next action by eps-greedy method
  - Input: 
    - q: a dictionary for value-action function
    - n0: a constant for deciding eps
    - ns: a dictionary counting the visiting times for states
    - state: the current state
  - Output:
    - action: the next chosen action
  '''
  # choose an action
  eps = 1. * n0 / (n0 + ns[state])
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
    a, b = q.get((state, 'h'), 0), q.get((state, 's'), 0)
    if a > b:
      action = 'h' 
    else:
      action = 's' 
  return action 

def TD(lambdas, k): 
  ''' 
  Sarsa(lambda) control with time-varying step size and eps-greedy exploration.
  - Input:
    - lambdas: a list of lambda values
    - k: total number of iterations
  - Output:
    - qs: a list of value-action functions for distinct lambdas 
      {0, 0.1, 0.2, ..., 1}.
  '''
  qs = list()
  n0 = 100 
  ns = dict()
  nsa = dict()
  for lam in lambdas:
    q = dict() # value-action function
    for _ in range(k):
      et = dict() # eligibility traces
      d, p = random.randint(1, 10), random.randint(1, 10) 
      r = 0 
      nexts = (d, p)
      ns.setdefault(nexts, 0)
      ns[nexts] += 1
      nexta = chooseAction(q, n0, ns, nexts)
      while d != 0 and p != 0:
        state, action = nexts, nexta
        r, nexts = step(state, action)
        d, p = nexts
        q.setdefault((state, action), 0)
        if d != 0 and p != 0:
          ns.setdefault(nexts, 0)
          ns[nexts] += 1
          nexta = chooseAction(q, n0, ns, nexts)
          q.setdefault((nexts, nexta), 0)
          err = r + q[(nexts, nexta)] - q[(state, action)]
        else:
          err = r - q[(state, action)] 
        et.setdefault((state, action), 0)
        et[(state, action)] = lam * et[(state, action)] + 1 
        for s, a in et: 
          q[(s, a)] += 1. / ns[s] * et[(s, a)] * err 
    
    qs.append(q)
  return qs

def MeanSquareError(qs, q): 
  ''' 
  Compute the mean-squared error between each function in qs and funciton q.
  - Input:
    - qs: a list of value-action functions
    - q: a dictionary for the target value-action function
  - Output:
    - errs: a list of mean-squared error for each function in qs and q.
  '''
  errs = list()
  for p in qs: 
    err = 0 
    for s, a in q:
      err += (p.get((s, a), 0) - q[(s, a)]) ** 2
    errs.append(err)

  return errs 

'''
# compute the optimal value-action function
q = MonteCarlo(6000000) 
with open('optimal-Q', 'w') as file1:
  for s, a in q:
    x, y = s
    print >>file1, x, y, a, q[(s, a)]

'''

# read the optimal value function
q = dict()
with open('optimal-Q', 'r') as file1:
  for l in file1:
    nums = l.split()
    s = (int(nums[0]), int(nums[1]))
    q[(s, nums[2])] = float(nums[3])


lambdas = [i * 0.1 for i in range(11)]
qs = TD(lambdas, 5000) 
errs = MeanSquareError(qs, q)
with open('lambdas-errors', 'w') as file2:
  for i in range(11):
    print >>file2, i * 0.1, errs[i]

lambdas = [0, 1]
errs = []
for i in range(20):
  qs = TD(lambdas, (i+1) * 5000)
  # each tuple in the list errs contains the episode number and errors for the functions
  errs.append(((i+1)*5000, MeanSquareError(qs, q)))

with open('learning-curve', 'w') as file3:
  for num, err in errs:
    print >>file3, num, err[0], err[1]

