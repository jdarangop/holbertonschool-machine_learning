# 0x01. Deep Q-learning

### 0. Breakout

Write a python script train.py that utilizes keras, keras-rl, and gym to train an agent that can play Ataris Breakout:

	- Your script should utilize keras-rls DQNAgent, SequentialMemory, and EpsGreedyQPolicy

	- Your script should save the final policy network as policy.h5

Write a python script play.py that can display a game played by the agent trained by train.py:

	- Your script should load the policy network saved in policy.h5

	- Your agent should use the GreedyQPolicy
