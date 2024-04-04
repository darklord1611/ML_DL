## Reinforcement Learning

- reward function
- state s
- application:
  - robot control
  - finance trading
  - factory optimization
  - video games
- an agent in an environment, in a state $s_{t}$ make an action $a_{t}$ -> get a reward $r_{t}$ and transit to the next state $s_{t}$
- Markov Decision Process(MDP) characterizes the state of the world completely
- defined by $(S,A,R,P,\gamma)$
  - $S$: set of possible states
  - $A$: set of possible actions
  - $R$: distribution of reward given for (state, action) pair
  - $P$: transition probability(or distribution) over the next given (state, action) pair
  - $\gamma$: discount factor(the longer the steps the smaller the reward)
### Properties of RL
- the return aka sum of rewards with discount factor that the system gets with certain actions
- goal? find policy $\pi$ that maps any state to an action that will maximize the return
    - example: chess
    - state: position of pieces on board
    - action: any legal moves
    - reward: +1 -> win, -1 -> lose, 0 -> tie
    - discount factor: assume it will be 0.995
    - return: $R = R_{1} + R_{2}\gamma + R_{3}\gamma^{2} + ....$
    - policy: find $\pi(s) = a$
    - first step is current state

- random policy vs optimal policy(think of searching problem in a grid)
- value function: evaluate how good is a state by calculating expected cumulative reward
- Q-value function(also known as $Q^{*}$): evaluate how good is a (state, action) pair by calculating expected cumulative reward
- (Bellman equation) as an iterative update -> hugh state space -> impossible
- function estimator to estimate the action-value function, if it is a NN -> deep Q-learning
