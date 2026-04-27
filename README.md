# Physics-Informed Neural Network (PINN) for Ball Trajectory Prediction

## 📌 Project Overview
This repository contains a implementation of a **Physics-Informed Neural Network (PINN)** designed to model the trajectory of an object in free fall. Unlike traditional "black-box" neural networks that rely solely on empirical data, this model integrates governing physical laws (Newtonian Mechanics) into the learning process.

By constraining the network with the fundamental laws of motion, the model achieves higher accuracy, better generalization, and robustness against sensor noise.

---

## 🧪 The Problem: Data Noise vs. Physical Reality
In real-world experimental scenarios, data is often "noisy" due to sensor inaccuracies or environmental interference. 
* **Standard Neural Networks:** Often treat noise as a feature, leading to **overfitting**. They learn the fluctuations rather than the underlying physics.
* **PINNs:** Treat physical laws as a regularizer. The network is trained not just to "match the dots," but to ensure that the predicted values satisfy the underlying differential equations.

---

## 🧮 Mathematical Foundation

The vertical position $h(t)$ of a ball thrown upwards is governed by the kinematic equation:
$$h(t) = h_0 + v_0 t - \frac{1}{2} g t^2$$

To inform the neural network, we utilize the first-order Ordinary Differential Equation (ODE) representing velocity:
$$\frac{dh}{dt} = v_0 - gt$$

### The PINN Loss Function
The model optimizes a composite loss function $\mathcal{L}_{total}$ to balance data fidelity with physical consistency:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{physics} + \lambda_3 \mathcal{L}_{IC}$$

1. **Data Loss ($\mathcal{L}_{data}$):** Mean Squared Error (MSE) between the predicted height and the noisy experimental data points.
2. **Physics Loss ($\mathcal{L}_{physics}$):** The MSE between the network's automatic differentiation output $\frac{d\hat{h}}{dt}$ and the theoretical velocity $(v_0 - gt)$.
3. **Initial Condition Loss ($\mathcal{L}_{IC}$):** Ensures the model respects the known starting position $h(0) = h_0$.

---

## 🛠 Technical Implementation

### Architecture
- **Type:** Multi-Layer Perceptron (MLP)
- **Hidden Layers:** 2 layers with 20 neurons each.
- **Activation Function:** `Tanh` (chosen for its non-zero second derivative, essential for higher-order ODEs).
- **Optimizer:** Adam Optimizer ($lr=0.001$).

### Key Features
* **Automatic Differentiation:** Leverages PyTorch’s `autograd` to compute derivatives of the output with respect to the input time $t$.
* **Collocation Points:** The physics loss is evaluated on 100 points across the time domain, even where experimental data is unavailable.
* **Hyperparameter Weighting:** The $\lambda$ coefficients are tuned to prioritize physical consistency ($\lambda_{ode} = 10.0$) over fitting noisy data ($\lambda_{data} = 5.0$).

---

## 🚀 Advantages
* **Noise Resilience:** Effectively filters out random fluctuations in experimental datasets.
* **Physical Consistency:** Prevents the model from making "impossible" predictions that violate gravity.
* **Data Efficiency:** Requires significantly fewer data points to reach convergence compared to standard ANNs because the physics "fills in the gaps."

---

## 📊 Results Visualization
The model successfully ignores the outliers in the red (noisy) data points and aligns closely with the black (exact) physical solution.
