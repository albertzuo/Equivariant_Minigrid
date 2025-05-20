# Zero-Shot Rotation Generalization in RL: Equivariance vs. Augmentation

## Project Overview

This project investigates methods to improve the generalization capabilities of Deep Reinforcement Learning (RL) agents, specifically their ability to adapt to environments visually transformed in ways not seen during training. We developed and evaluated a $C_4$-equivariant convolutional neural network that inherently understands rotational symmetry. The key result is that this equivariant agent demonstrated robust zero-shot generalization to unseen spatial rotations of its environment, significantly outperforming standard convolutional architectures and data augmentation techniques under the same conditions.

## Background: The Generalization Problem in RL

Deep RL agents often exhibit a tendency to overfit to the specific characteristics of their training environments. This limits their performance when deployed in new or slightly modified situations, a common challenge in real-world applications. Improving generalization typically requires agents to learn more robust representations that are invariant or equivariant to irrelevant transformations in the environment.

## Experimental Setup

The study was conducted in a modified MiniGrid-FourRooms-v0 environment, a grid-based navigation task where the agent must reach a target. Observations were symbolic, top-down views of the grid.

Three types of agents were trained using Proximal Policy Optimization (PPO):
1.  **Standard CNN Agent:** A baseline convolutional neural network.
2.  **Data-Augmented CNN Agent:** An identical CNN architecture trained with observations randomly rotated by $0^{\circ}$ or $90^{\circ}$ during each episode.
3.  **$C_4$-Equivariant Agent:** A convolutional network incorporating $C_4$ rotational equivariance using group-equivariant layers. This agent was trained only on the environment in its standard $0^{\circ}$ orientation.


The Standard CNN Agent and the $C_4$-Equivariant Agent were trained on the standard orientation environment. Generalization was evaluated zero-shot – without any further training or fine-tuning – on environment observations rotated by $90^{\circ}$, $180^{\circ}$, and $270^{\circ}$.

## Key Results

* The **Fully $C_4$-Equivariant Agent** demonstrated strong generalization, achieving success rates of approximately 90% across all tested rotations ($0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ}$), despite only training on the $0^{\circ}$ orientation.
* The **Standard CNN Agent** failed to generalize to any of the unseen rotations ($90^{\circ}, 180^{\circ}, 270^{\circ}$).
* The **Data-Augmented Agent** succeeds on the $90^{\circ}$ rotation (which was included in its augmented training data) but failed to generalize to the entirely unseen $180^{\circ}$ and $270^{\circ}$ rotations.

These findings highlight that architectural inductive biases, such as equivariance, can be more effective for certain types of generalization than data augmentation alone.


### No Rotation




https://github.com/user-attachments/assets/2977a128-36bc-4889-bcd0-4f99a4e76cd9




### $90^{\circ}$ Rotation




https://github.com/user-attachments/assets/bfe4f460-c2da-491f-a398-b98b34e29a93




### $180^{\circ}$ Rotation



https://github.com/user-attachments/assets/a6eba20d-3e78-4051-bb46-eda2178d60c8





### $270^{\circ}$ Rotation




<V




