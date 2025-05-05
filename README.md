# A Physics-Informed In-Context Learning Framework for Online Interaction Prediction in Robotic Tasks

These are the Python scripts and datasets to train and reproduce the results of the paper "A Physics-Informed In-Context Learning Framework for Online Interaction Prediction in Robotic Tasks". This is an anomyzied github for submission purposes. The introduced approach extends transformer-based meta learning with physically grounded inductive biased, including learnable physics parameters, physics-aware embeddings, and regularization via physics-basedd loss function. The model is trained on real-world interaction datasets collected from a robotic arm executing chirp-like trajectories against different surfaces.

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

## [Main paper training](./main_paper_codes/)


  - [Physics-informed model training](./main_paper_codes/InteractionMetaModel_Physics_train.py),
  - [Data-driven model training](./main_paper_codes/InteractionMetaModel_Data_train.py)


Run any of the codes with the corresponding [datasets](./Datasets/) for the training, which will save a model 'Interaction_metamodel_physics.pth' or 'Interaction_metamodel_data.pth', for the physics-informed or data-driven models, respectively, every 200 epochs. 

Please note that the models will try to utilize "cuda" if available, if not, the training process may be slow.

# Managing evaluation

## [Main paper evaluation](./main_paper_codes/)

- [Physics-informed model testing](./main_paper_codes/Test_interactionModel_Physics.py)
- [Data-driven model testing](./main_paper_codes/Test_interactionModel_Physics.py) (still to be fixed)

You can run the evaluation of both physics-informed or data-driven models, on any of the [datasets](./Datasets/), after at least training for 200 epochs, so you may have a saved checkpoint.



