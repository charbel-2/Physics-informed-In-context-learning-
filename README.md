# A Physics-Informed In-Context Learning Framework for Online Interaction Prediction in Robotic Tasks

This is an anomyzied github for the paper "A Physics-Informed In-Context Learning Framework for Online Interaction Prediction in Robotic Tasks".

![Figure 1: System architecture diagram](Images/Model_scheme.png)

This framework has been applied to the identification of interaction forces between a robotic arm and its environment. The proposed physics-informed in-context learning solution showed improved generalization when interacting with different environments, of different materials.

**Inputs**:
  -Cartesian positions (x, y, z),
  -Cartesian  velocities (\dot{x}, \dot{y}, \dot{z}),
  -Cartesian accelerations (\ddot{x}, \ddot{y}, \ddot{z}), 
  - Cartesian target positions (x_T, y_T, z_T),
  - Cartesian velocities (\dot{x}_T, \dot{y}_T, \dot{z}_T)
  - **Velocities**: (ẋ, ẏ, ż)
  - **Accelerations**: (ẍ, ÿ, z̈)
  - **Target positions**: (xₜ, yₜ, zₜ)
  - **Target velocities**: (ẋₜ, ẏₜ, żₜ)
