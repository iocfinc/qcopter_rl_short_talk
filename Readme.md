# Quadcopter Project: Udacity project on RL

<p align="center">
  <img width="750" height="450" src="https://raw.githubusercontent.com/craig-martinson/quadcopter-project/master/movie.gif"><br>
  This is what is supposed to happen (not my results though 🙃)
</p>

## Introduction:

As part of a previous nanodegree on Deep Learning, we got to implement any RL model of our choice to control a simulated quadcopter. It was up to us to determine the task we want, for example: lifting off, loitering, moving to a position. We got to define our reward function for the task, more on this later. We also got to read through and decide which learner to use.

## Problem breakdown

* __*Action space*__: discrete/**continous**
Working on 4-dimmensional action space (the individual propellers) and 2 actions (high and low rotation speed).

* __*State-space*__: discrete/**continous**
In this task, we are working on a 6-dimensional state space. These correspond to the position and linear velocity, a total of 6 because we work on three axes x,y,z.

* __*Environment*__:
In this project, the environment comes in the form of a Physics simulator. The simulator takes in the action space inputs (rotor velocities) and translates it to the *next_iteration* state space. Looking deeper into the physics sim, we will see that it has calculations for `linear_forces`, `moments`, `propeller_thrust`, and `propeller_wind_speed`, among other things.

* __*Learning Algorithm*__:
Used Deep Deterministic Policy Gradient (paper in the reference) if you want to learn more. Conviniently forgot about this 😁

* __*Tasks*__:
In this project, the task crafted was to get the quadcopter to move to a defined position and hover/loiter. In the GIF above, the task is to get the quadcopter to lift-off and hover at a given position.

```python
init_pose = np.array([50., 50., 50., 0., 0., 0.])                # Initialization:  Position on X,Y,Z  
init_velocities = np.array([100., 100., 100.])                   # Initialization:  Velocities on X,Y,Z (will the quadcopter move along X,Y,Z)
init_angle_velocities = np.array([0., 0., 0.])                   # Initialization:  Angular Velocities on X,Y,Z (will the quadcopter be spinning # note different from the rotors spinnning)

num_episodes = 1000                                              # Initialization:  Number of trials
target_pos = np.array([150., 150., 150.])                        # Initialization:  Target position for the task on X,Y,Z
task = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=target_pos)      # Define the task for the PhysicsSim
agent = DDPG(task)        # Pass the task to the DDPG agent
```

In the initialized `init_pose` and `target_pos` above, you can visualize the task as moving from one corner of the room to another.

## Putting it together

<p align="center">
  <img width="640" height="360" src="https://miro.medium.com/max/1400/0*P8_XhjgYI-sDsygl"><br>
  <b></b>
</p>

The primary explanation of RL would have to be learning through experience.

> An Agent **learns** by continually interacting with its environment with the objective of *maximizing* a reward/goal.

With the physics sim provided, we can get the **current state**, **action**, and a **simulated next state**. In this case, the states are the position and velocities of the quadcopter. The objective is to get to a defined position. The question now would be *How do we know if we are improving or not?*

## Reward Shaping

In this project the final reward function was given as:

```python
gap_ = np.sqrt((abs(self.sim.pose[:3] - self.target_pos[:])**2).sum())                        # Resultant gap between the target and pos.
res_vel = np.sqrt((abs(self.sim.v[:])**2).sum())                                              # Resultant velocity of the quadcopter.
reward = -10*np.tanh(.01*gap_) + 0.001*res_vel + (1000*np.tanh(1/(0.00001+abs(self.sim.pose[:3]-self.target_pos[:])))).sum()
```

### **Response curve of the reward components**
<p align="center">
  <img width="750" height="750" src="./6232020-Gap distance response.png"><br>
  <b>The intuition here is that we want to close the gap so that we **minimize** the punishment from having a significant gap in distance between the current position and desired position.</b>
</p>

<p align="center">
  <img width="750" height="750" src="./6232020-vel response.png"><br>
  <b>For velocity, we want it to be high such that the agent would favour action vs inaction (~0 velocities).</b>
</p>

<p align="center">
  <img width="750" height="750" src="./6232020-Pose response.png"><br>
  <b>Finally, we want to maximize the reward by minimizing the absolute difference between the actual position and the desired position. It works in conjunction with Gap Distance.</b>
</p>

Based on the reward function, we can say that the agent aims to move (velocity component) to the desired position (- gap) and maintain that position (Pose). Also, we can see that there are differences in weightage of the different components.


## Some lessons learned

### **Hacking through overfitting**

The possibility of overfitting on the environment is doable. Notice the reward functions above are quite tailored. The good thing with RL is that overfitting is inherent in the definition:**MAXIMIZE a reward/goal**.  By this framework,  greedy algorithms work great because it would want to maximize outcomes. Greedy could lead to an imperfect understanding of the environment. The trail is leading us to the exploration and exploitation problem, which we will not cover. 

> Imagine having to choose between 2 doors for a prize: For trial one we choose door B and get a reward of 0. For trial two, door A and get a reward of 10. If we apply just a greedy approach, "for all succeeding trials choose door A". What if door B give a reward by the function: 10% chance of Reward = 1e6 else Reward = 0 😶? If this was a continous task then which would have been the better choice?

### __RL **can** be sample intensive__

Since RL is trying to model the world through interactions, it might be necessary to train over large volumes of samples to get a good idea of the environment. Building on the **Hacking through overfitting** discussion, we know that to counter it would mean increasing the probability of exploration vs exploitation. To have an idea of **what to do** we have to experience examples of **what not to do**. Depending on how large or small the environment is and how much degrees of freedom the agent has, it can require millions of samples to achieve human-level performance.

### **You get what you rewarded**

One thing with RL is that the reward function has to be succinct enough that it influences the agent on the steps involved and not too sparse that weird behaviours start to come out. It would depend on whether it in the context of a Research perspective or Application/output context. RL agents can come up with some unique way to solve things. The uniqueness of the solution is excellent if the context was seeing new ways to solve a task was the objective. For the project, where we have an idea of the task at hand, which means that we have expected behaviour to take shape. For example, in the reward function presented above, the second term 0.001*res_vel was added to ensure that the agent prefers motion. When we talk about quadcopters and how turning on the propellers would provide lift it sounds obvious. For an agent to learn it, it would take time (if it even learns it at all). Which is why defining the expected behaviour to fit the task is essential. 

>A researcher gives a talk about using RL to train a simulated robot hand to pick up a hammer and hammer in a nail. Initially, the reward was defined by how far the nail was pushed into the hole. Instead of picking up the hammer, the robot used its own limbs to punch the nail in. So, they added a reward term to encourage picking up the hammer, and retrained the policy. They got the policy to pick up the hammer…but then it threw the hammer at the nail instead of actually using it. - [Alexirpan post](https://www.alexirpan.com/2018/02/14/rl-hard.html)

### **Positive and Negative framing has implications**

If a person receives punishment for doing something, the way to maximize reward would be to do less of the action that led to the punishment. Positive rewards, on the other hand, forces us to work to improve and get more of the positive reward. It directly maps to how humans tend to behave. If the environment is so toxic and negative, then it is natural to aim to end it as early as possible. Since the goal is to maximize reward, and all rewards are negative, then we work towards getting zero rewards to ensure maximum reward.

The behaviour changes if we change the framing of the reward to a positive one. Instead of aiming to end it things early, by maximizing reward, we would want to continue to work longer. Meaning we would work towards improving and keeping the task going.

> Make fewer mistakes vs increasing correct answers mean the same thing as getting higher marks. The difference would be on framing. Make fewer mistakes mean punishment for every wrong answer. Getting more correct answers would mean an additional reward for every correct answer. Its the difference of feeling the pressure for every mistake made and being happy for every correct answer made. Which would be better? 🤔

It depends on framing a task as episodic or continuous. If it is episodic, then a negative frame would be better since it ends the episode fast. If it is continuous, then the positive frame would suit it better since it forces the task to continue for as long as possible.

### **Simulations are simulations**

There is a gap in training using simulation and training in the real world. In a simulation, we can control the variables. In this case, the quadcopter will start to learn because it can map out that increasing velocity will lift it up. Providing bias to the propeller rotation will influence its direction of travel (forward, backwards, left, right). Since this is in a physics sim, we have a reasonably good idea of what the next state is going to be. Once it onboards to real hardware, it becomes a whole other thing. If the wind blows for example and the quadcopter drifts, will it map that drift (which was external) to the result of the action taken? If the battery was low leading to a lower propeller power and lower lift, will the agent figure out what to do and not crash?

### **Tasks are not what they seem**

<p align="center">
  <img width="450" height="450" src="https://media-exp1.licdn.com/dms/image/C5622AQGxIPo4jgW3nw/feedshare-shrink_800/0?e=1596067200&v=beta&t=lcUGd4eWuaESxDjw-CwKe02BcYPnfqOUCOJ__T5rcXo"><br>
</p>

<p align="center">
  <img width="450" height="450" src="https://greydanus.github.io/assets/visualize-atari/breakout-v0.gif"><br>
</p>

An agent learning to play Breakout in Atari sounds cool. It sounds way cooler than say drinking water out of a glass.  We tend to take for granted how complex drinking water is because it is trivial for us. In Breakout, the action space is minimal (move left or right). The objective is to maximize the score. For the drinking water task, it sounds simple, but many tasks are involved. Grip the glass, do not break the glass, raise the glass without pouring it, tilt the glass slowly, so it does not spill.

## Go and try it out

Do not trust me on these things. Apply RL to learning RL. Try it out yourself and come up with your notions of what RL is. Interact with it, build things out of it and see what lessons you can learn. Its what makes it fun, your state and environment interacting with RL would be different from mine so new ideas may come to you. 😉

<p align="center">
  <img width="450" height="450" src="https://dudeperf3ct.github.io/images/series_rl/rl_meme.jpg"><br>
</p>

## Resources and References:

[Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)<br>
[SpinningUp OpenAI DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html#:~:text=Deep%20Deterministic%20Policy%20Gradient%20(DDPG,function%20to%20learn%20the%20policy))<br>
[Sutton and Barto RL Book](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)<br>
[alexirpan on RL's promise](https://www.alexirpan.com/2018/02/14/rl-hard.html)<br>
[Faulty Reward Functions in the Wild](https://openai.com/blog/faulty-reward-functions/)<br>
[Optimization-based locomotion planning, estimation, and control design for the atlas humanoid robot: Atlas @ Boston Dynamics](https://dspace.mit.edu/handle/1721.1/110533)<br>
[Python Robotics Repo](https://github.com/AtsushiSakai/PythonRobotics)<br>

