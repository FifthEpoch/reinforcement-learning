# Reinforcement Learning Robotics

Reinforcement learning (RL) enables a robot to autonomously discover an optimal behavior through trial-and-error interactions with its environment. This is an experiment with the goal of observing the difference in the agent's learning between simulated environment and the real world.

This is an ongoing project, new components will be added as improvements are needed in the experiment. 


## Goal

The learning goal of the robotic agent is to regulate the reading from the on-board temperature sensor within a certain window (in the code, that window is currently set to a half degree Celsius window of 25.7°C - 26.2°C).

The robotic agent also should learn to avoid head-on collision when regulating temperature. 


## Hardware Setup

<img src="https://github.com/FifthEpoch/reinforcement-learning/blob/main/RL-robotics/img/robot-top.jpeg" width="400" height="auto"> <img src="https://github.com/FifthEpoch/reinforcement-learning/blob/main/RL-robotics/img/robot-side.jpeg" width="400" height="auto">

Our agent is a small computer (Raspberry Pi 3 Model B) driving a microcontroller (Arduino Uno). The microcontroller is connected to an array of sensors, as well as a pair of DC motors which provide mobility to the agent. The onboard computer is responsible for computation of action-values, storing learning, and instructing the microcontroller to take an action on each time step. The microcontroller is responsible in collecting sensory data and executing actions instructed by the computer (i.e. turning in a certain direction, driving motors to move forward or backward).

**Sensors Used**

* HC-SR04 Ultrasonic Sensor
* DS18B20 Temperature Sensor
* D2SW-P2L2 Hinge Roller Lever Basic Switch x 2
* NPN Phototransistor x 2


## Environment

<img src="https://github.com/FifthEpoch/reinforcement-learning/blob/main/RL-robotics/img/arena.jpeg" width="400" height="auto">
![Arena for experiment](https://github.com/FifthEpoch/reinforcement-learning/blob/main/RL-robotics/img/arena.jpeg)

An hexagonal, walled enclosure was built to contain the robotic agent to an area, the walls ensure that the switches at the front of the robot is triggered whenever there is a collision. The enclosure is about 8 feet wide corner to corner. A heat lamp radiates light and heat towards the center of the enclosure.


## Software

We are using a Python program ran on the Raspberry Pi to communicate to the microcontroller via serial  and to process incoming sensory data. For the microcontroller, we use a sketch written in Arduino’s standard language C++. From here on, we will refer to the Raspberry Pi as the *computer*, and Arduino as the *microcontroller*.

For the Reinforcement Learning algorithm, we have 2 approaches: 
1. **Sarsa(λ)**: a value-based function approximation method with eligibility trace
2. **Actor-Critic**: a policy-based method with eligibility trace

Sensory data are used to determine which features are active in the feature vector (**x**). To do this, we apply state aggregation to reduce the state space. For example, temperature readings are broken down into 15 states, the first state includes any readings from 20.0°C to less than 20.67°C. We have also included a rate of change feature for the phototransistors and temperature sensors. The rate of change is calculated by comparing the current reading with readings collected from the last 3 time steps for Sarsa(λ) and last 1 time step for Actor-Critic.

Generalization is also applied using tile coding, 4 tilings are used and feedback data are shifted by one unit randomly to generate 4 active features per time step.

A reward of -1 is assigned whenever temperature reading is lower than 25 °C or higher than 26 °C, while the reward of +1 is assigned when temperature reading is higher or equal to 25 °C or lower or equal to 26 °C. Additionally, a reward of -3 is assigned when the left or the right bump switches are triggered in a collision. 

**Library Used**

* PySerial
* NumPy
* Math (python module)
* Servo.h
* OneWire.h
* DallasTemperature.h

## Next Steps

The first 2 hours of the Sarsa(λ) implementation has been condensed into a 2-minute video available at https://youtu.be/qc30KshEg2g

Further improvements are needed for the training process and in feature shaping: 

* build a simulated environment using data collected in real-life to speed up training
* experiment on tracking robot's location within the arena and using the coordinates as features instead of using sensory input as features


Thank you for checking out this project, more to come. 
