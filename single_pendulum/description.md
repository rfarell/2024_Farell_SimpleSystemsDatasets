# Symplectic Spectrum Gaussian Processes (2022 Tanaka, Iwata & Ueda)


## Part 1: Description of the System

### Virtual Model of Pendulum System

The system under consideration is a simple pendulum. The schematic provided below describes its configuration. Key parameters of the pendulum include:

- **Length of the string, $l$** (in meters): This is the distance from the pivot point to the mass.
- **Mass at the end of the string, $m$** (in kilograms): This is the weight suspended by the string.
- **Gravitational constant, $g$** (in meters per second squared): This is the acceleration due to Earth's gravity, affecting the pendulum's motion.

The state of the pendulum is defined by the angle $\theta$, which measures the deviation of the string from the vertical axis, and the conjugate momenta $p_{\theta}$, representing the momentum associated with the angular position.

<div style="text-align: center;">
    <img src="system_schematic.png" width="30%" alt="Schematic of the Pendulum">
</div>

#### Forming the Lagrangian

The Lagrangian $\mathcal{L}$ of a system is a function that summarizes the dynamics of the system. It is defined as the difference between the kinetic energy $T$ and the potential energy $V$ of the system:
$$
\mathcal{L} = T - V
$$

##### Kinetic Energy $T$
For a simple pendulum, which consists of a mass $m$ at the end of a rod of length $l$ swinging under gravity, the kinetic energy is derived from the motion of the mass along a circular path. The velocity $v$ of the mass is given by $l\dot{\theta}$, where $\dot{\theta}$ is the angular velocity. Hence, the kinetic energy $T$ is:
$$
T = \frac{1}{2} m v^2 = \frac{1}{2} m (l\dot{\theta})^2 = \frac{1}{2} ml^2\dot{\theta}^2
$$

##### Potential Energy $V$
The potential energy $V$ is due to the height of the mass from a reference level, which we can take as the lowest point of the pendulum's path. If $\theta = 0$ represents the lowest point, the height $h$ of the mass is $l - l\cos(\theta)$. Thus, the potential energy $V$ due to gravity is:
$$
V = mgh = mg(l - l\cos(\theta)) = mgl(1 - \cos(\theta))
$$

##### The Lagrangian
By combining these expressions for $T$ and $V$, the Lagrangian $\mathcal{L}$ is:
$$
\mathcal{L} = \frac{1}{2}ml^2\dot{\theta}^2 - mgl(1 - \cos\theta)
$$

#### Deriving the Hamiltonian

The Hamiltonian $\mathcal{H}$ is derived from the Lagrangian through a transformation involving the conjugate momentum. The conjugate momentum $p_\theta$ corresponding to the coordinate $\theta$ is defined as:
$$
p_{\theta} = \frac{\partial\mathcal{L}}{\partial\dot{\theta}} = ml^2\dot{\theta}
$$

##### Hamiltonian $\mathcal{H}$
The Hamiltonian $\mathcal{H}$ is given by:
$$
\mathcal{H} = \sum_{\theta} \dot{\theta}p_{\theta} - \mathcal{L}
$$

For the simple pendulum with only one generalized coordinate $\theta$, this becomes:
$$
\mathcal{H} = \dot{\theta}p_{\theta} - \mathcal{L}
$$
Substituting the expressions for $\mathcal{L}$ and $p_{\theta}$, and rewriting $\dot{\theta}$ in terms of $p_{\theta}$:
$$
\mathcal{H} = \dot{\theta}(ml^2\dot{\theta}) - \left(\frac{1}{2}ml^2\dot{\theta}^2 - mgl(1 - \cos\theta)\right)
$$
$$
\mathcal{H} = \frac{p_{\theta}^2}{2ml^2} + mgl(1 - \cos\theta)
$$

The formulation and transformation from the Lagrangian to the Hamiltonian are thus complete. This transition provides a powerful way to analyze the system in terms of energy rather than forces, which can be particularly useful in complex systems and in the field of quantum mechanics.

#### Simple Demo of the Pendulum System and its Phase Space
In our example, we will fix $m=1kg$, $l=1m$, and $g=3m/s^2$. 

This gives us the hamiltonian
$$
H(\theta, p_{\theta}) = \frac{p_{\theta}^2}{2} + 3(1 - \cos\theta).
$$
Some example trajectories are shown below.
<div style="text-align: center;">
    <img src="pendulum_simulation_with_phase_space.gif" width="100%" alt="Schematic of the Pendulum Simulation">
</div>

### Problem Objective

The goal of this problem is to infer the unknown Hamiltonian of a pendulum system using observational data collected through a series of experiments. In each experiment, a lab technician sets a pendulum at a specified initial angle $\theta$ and imparts a specific momentum $p_{\theta}$. The pendulum is then released and its motion is recorded for $T$ seconds, capturing $J$ frames at regular intervals. These frames are used to estimate the state $(\theta, p_{\theta})$ at each point, albeit with some noise introduced by the limitations of camera-based measurements.

Our objective is to develop a probabilistic model that accurately represents the Hamiltonian function of the pendulum, which depends on $(\theta, p_{\theta})$. This model is derived directly from the noisy data of $I$ experiments. Upon successful training, the model’s efficacy is verified by predicting the pendulum’s behavior under new, untested initial conditions.

**Evaluation of Model Performance**

The effectiveness of the model is determined by its ability to predict future states of the pendulum given some initial conditions. Specifically, a well-performing model should satisfy the following criteria:

1. **Calibration:** The model should provide well-calibrated predictions. For a predicted distribution $D(\theta, p_{\theta})$, empirical validation should show that the probability $p(\theta_L \leq \theta \leq \theta_U, p_{\theta,L} \leq p_{\theta} \leq p_{\theta,U})$ closely matches the observed frequency $\frac{1}{N} \sum_{i=1}^N \mathbb{1}(\theta_L \leq \tilde{\theta} \leq \theta_U, p_{\theta,L} \leq \tilde{p_{\theta}} \leq p_{\theta,U})$, where $(\tilde{\theta}, \tilde{p_{\theta}})$ are the test observations.

2. **Variance Reduction:** The variance of the predicted distribution $D(\theta, p_{\theta})$ should be minimized, indicating precise and consistent predictions.

#### Why are we learning the Hamiltonian?
We're doing this because we can use the Hamiltonian to predict the future state of the pendulum system given some initial conditions. Consider the generalized coordinates $\mathbf{x}=(\mathbf{x}_q, \mathbf{x}_p)$. Then Hamiltonian dynamics tells us that:
$$
\begin{align*}
\frac{d \mathbf{x}}{d t} &= (\mathbf{S}-\mathbf{R}) \nabla H(\mathbf{x}) =: \mathbf{f}(\mathbf{x}), \quad \text{where} \quad \mathbf{S} = \begin{pmatrix} \mathbf{0} & \mathbf{I} \\ -\mathbf{I} & \mathbf{0} \end{pmatrix}\\
\end{align*}
$$
where $\mathbf{R}$ is a positive semi-definite dissipation matrix. When $\mathbf{R}=\boldsymbol{0}$, the vector field conserves total energy and has a symplectic geometric structure. The form of $\mathbf{R}$ below representes a dissipation system with friction coefficients $r_i \ge 0$:
$$
\mathbf{R} = \begin{pmatrix} \mathbf{0} & \mathbf{0} \\ \mathbf{0} & diag(\mathbf{r}) \end{pmatrix}
$$


After simplifying the above, we get the dynamics of the generalized coordinates:
$$
\begin{align*}
 \frac{d \mathbf{x}_q}{d t} &= \nabla_{\mathbf{x}_p}H(\mathbf{x}) \\
\frac{d \mathbf{x}_p}{d t} &= -\nabla_{\mathbf{x}_q}H(\mathbf{x}) - diag(\boldsymbol{r})\cdot \nabla_{\mathbf{x}_p}H(\mathbf{x})
\end{align*}
$$

We can use these to predict the state of the system at any future point in time:
$$
\begin{align*}
\mathbf{x}(t)&=\mathbf{x}_{1}+\int_{t_{1}}^{t} \mathbf{f}(\mathbf{x}) d t.
\end{align*}
$$

**EXAMPLE: PENDULUM SYSTEM**
For the pendulum system with friction coefficient $r$, this is
$$
\begin{align*}
\begin{pmatrix} \frac{d \theta}{d t} \\ \frac{d p_{\theta}}{d t} \end{pmatrix} &= \begin{pmatrix} 0 & 1 \\ -1 & -r \end{pmatrix} \cdot \begin{pmatrix} \frac{\partial H(\theta,p_{\theta})}{\partial \theta} \\ \frac{\partial H(\theta,p_{\theta})}{\partial p_{\theta}} \end{pmatrix} = \mathbf{f}(\theta, p_{\theta})\\
\Rightarrow \frac{d \theta}{d t} &= \frac{\partial H(\theta,p_{\theta})}{\partial p_{\theta}} \\
\Rightarrow \frac{d p_{\theta}}{d t} &= -\frac{\partial H(\theta,p_{\theta})}{\partial \theta} - r\frac{\partial H(\theta,p_{\theta})}{\partial p_{\theta}} \\
\Rightarrow \begin{pmatrix} \theta \\ p_{\theta} \end{pmatrix}(t) &=\begin{pmatrix} \theta \\ p_{\theta} \end{pmatrix}(t_0) +\int_{t_{0}}^{t} \mathbf{f}(\theta, p_{\theta}) d t
\end{align*}
$$

Using the form of pendulum systems Hamiltonian derived above
$$
\mathcal{H} = \frac{p_{\theta}^2}{2ml^2} + mgl(1 - \cos\theta)
$$
we can calcuate the state dynamics as 
$$
\begin{align*}
\frac{d \theta}{d t} &= \frac{\partial H(\theta,p_{\theta})}{\partial p_{\theta}} \\
&= \frac{p_{\theta}}{ml^2}\\
\frac{d p_{\theta}}{d t} &= -\frac{\partial H(\theta,p_{\theta})}{\partial \theta} - r\frac{\partial H(\theta,p_{\theta})}{\partial p_{\theta}} \\
&= -mgl \sin (\theta) - r\frac{p_{\theta}}{ml^2}
\end{align*}
$$

This gives us
$$
\begin{align*}
\mathbf{f}(\theta, p_{\theta}) = \begin{pmatrix}
\frac{p_{\theta}}{ml^2} \\ -mgl \sin (\theta) - r\frac{p_{\theta}}{ml^2}
\end{pmatrix}
\end{align*}
$$

### Real-World Dataset of Pendulum System

The dataset consists of $I$ noisy trajectories represented by the series $\left\{\left(t_{ij}, \mathbf{y}_{ij}\right)\right\}$, where $i=1, \ldots, I$ and $j=1, \ldots, J_i$. Each trajectory is made up of $J_i$ observations over time. We define $\mathbf{x} = (\theta, p_{\theta})$ as the state variables of the pendulum, where $\theta$ is the angle and $p_{\theta}$ is the conjugate momentum.

Each observed state $\mathbf{y}_{ij}$ is a noisy measurement of the true state $\mathbf{x}(t_{ij})$ at time $t_{ij}$, modeled as:

$$
\mathbf{y}_{ij} = \mathbf{x}(t_{ij}) + \epsilon_{ij},
$$

where $\epsilon_{ij}$ represents the observational noise. This noise is assumed to follow a specific known probability distribution, $p(\epsilon)$.

The dataset visualization below displays these trajectories for the entire training dataset. Colors in the plot distinguish individual trajectories from one another.

<div style="text-align: center;">
    <img src="pendulum_train_trajectories.png" width="100%" alt="Training Dataset">
</div>