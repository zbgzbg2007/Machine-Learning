import random

'''
This file implements the environment of the game Easy21.

The state consists of two numbers: the first number of dealer and the
sum of the player. If one goes bust, he gets 1 as result.
When one of the two numbers is 0, the state is terminal. When the game
is terminal but non of them is 0, the function step will compare the 
two numbers and return a trivial state that contains 0 for the lost part, 
or both 0 if draw.

The action could be h (for hit) and s (for stick): h will give 
another card to player and s will stop drawing for player. 

'''


def draw():
  '''
  Simulate the operation of drawing a card randomly.
  - Input:
    - None
  - Output: 
    - value of the card
  '''
  val = random.randint(1, 10)
  col = random.randint(1, 3)
  if col == 3:
     col = -1
  else:
     col = 1
  return val * col

def step(state, action):
  ''' Simulate one step of the game.
  - Input:
    - state: the state before the action
    - action: the action of the player: 'h' means hit and 's' means stick
  - Output:
    - r: reward of the action
    - state: the state after the action
  '''
  d, p = state
  r = 0
  if action == 'h':
    p += draw()
    if p > 21 or p < 1:
      p = 0
      while d < 17 and d > 1:
        d += draw()
      if d > 21 or d < 1:
        d = 0

  elif action == 's':
    while d < 17 and d >= 1:
      d += draw()
    if d > 21 or d < 1:
      d = 0
    if d == p:
      d, p = 0, 0
    elif d > p:
      p = 0
    else:
      d = 0
  else:
    print 'Wrong Action!!'
  if p == 0 or d == 0:
    if p > d:
       r = 1
    elif p < d:
       r = -1
  return r, (d, p)
