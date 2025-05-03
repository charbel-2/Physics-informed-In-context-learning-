# A Physics-Informed In-Context Learning Framework for Online Interaction Prediction in Robotic Tasks

This is an anomyzied github for the paper "A Physics-Informed In-Context Learning Framework for Online Interaction Prediction in Robotic Tasks".

![Figure 1: System architecture diagram](Images/Model_scheme.png)

This framework has been applied to the identification of interaction forces between a robotic arm and its environment. The proposed physics-informed in-context learning solution showed improved generalization when interacting with different environments, of different materials.

**Inputs**:


  - **Cartesian Positions**: (x, y, z),
  - **Cartesian Velocities**: (ẋ, ẏ, ż),
  - **Cartesian Accelerations**: (ẍ, ÿ, z̈),
  - **Cartesian Target positions**: (xₜ, yₜ, zₜ),
  - **Cartesian Target velocities**: (ẋₜ, ẏₜ, żₜ)

**Outputs**:


  - **Interaction Forces**: (Fₓ, Fᵧ, F_z)


This framework has been applied to Panda Franka Robotic Arm.

# Managing training

## [Main_paper_codes](./main_paper_codes/)
