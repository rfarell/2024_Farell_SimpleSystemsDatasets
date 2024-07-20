### Description of the Henon-Heiles System

#### Overview
The Henon-Heiles system is a well-known model in the study of nonlinear dynamics and chaos theory. Originally introduced to describe the motion of a star around a galactic center, it has since become a classic example for studying chaotic behavior in dynamical systems. The system consists of two degrees of freedom and is characterized by a specific potential energy function that leads to non-linear and complex motion.

#### Lagrangian Formulation
The Lagrangian formulation of the Henon-Heiles system describes the system in terms of kinetic and potential energy. The Lagrangian $L$ is given by:

$$L = T - V$$

where $T$ is the kinetic energy and $V$ is the potential energy.

1. **Kinetic Energy ($T$)**:
$$T = \frac{1}{2} \left( \dot{x}^2 + \dot{y}^2 \right)$$

2. **Potential Energy ($V$)**:
$$V = \frac{1}{2} \left( x^2 + y^2 \right) + x^2 y - \frac{1}{3} y^3$$

Combining these, the Lagrangian $L$ of the Henon-Heiles system is:
$$L = \frac{1}{2} \left( \dot{x}^2 + \dot{y}^2 \right) - \left( \frac{1}{2} \left( x^2 + y^2 \right) + x^2 y - \frac{1}{3} y^3 \right)$$

#### Hamiltonian Formulation
The Hamiltonian formulation is another powerful way to describe dynamical systems, particularly useful in the study of conservative systems. The Hamiltonian $H$ represents the total energy of the system and is expressed in terms of coordinates and conjugate momenta. For the Henon-Heiles system, the Hamiltonian $H$ is given by:

$$H = T + V$$

where $T$ is the kinetic energy and $V$ is the potential energy.

1. **Coordinates**:
$$q_1 = x, \quad q_2 = y$$

2. **Momenta**:
$$p_1 = \dot{x}, \quad p_2 = \dot{y}$$

3. **Hamiltonian ($H$)**:
$$H = \frac{1}{2} \left( p_1^2 + p_2^2 \right) + \frac{1}{2} \left( q_1^2 + q_2^2 \right) + q_1^2 q_2 - \frac{1}{3} q_2^3$$

Substituting the coordinates and momenta:
$$H = \frac{1}{2} \left( p_x^2 + p_y^2 \right) + \frac{1}{2} \left( x^2 + y^2 \right) + x^2 y - \frac{1}{3} y^3$$

#### Dynamics of the Henon-Heiles System
The equations of motion for the Henon-Heiles system can be derived using Hamilton's equations:

$$\dot{q_i} = \frac{\partial H}{\partial p_i}$$
$$\dot{p_i} = -\frac{\partial H}{\partial q_i}$$

For the Henon-Heiles system, the equations of motion are:
$$\dot{x} = \frac{\partial H}{\partial p_x} = p_x$$
$$\dot{y} = \frac{\partial H}{\partial p_y} = p_y$$
$$\dot{p_x} = -\frac{\partial H}{\partial x} = -x - 2xy$$
$$\dot{p_y} = -\frac{\partial H}{\partial y} = -y - x^2 + y^2$$

These equations describe the evolution of the system over time, revealing the complex and often chaotic trajectories that arise from the nonlinear interactions in the Henon-Heiles potential.
