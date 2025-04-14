from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
from sklearn.neural_network import MLPRegressor

# --- USER INPUT SECTION ---
# Get plant transfer function from user input
nums = [int(x) for x in input("Enter numerator: ").split()]
dens = [int(x) for x in input("Enter denominator: ").split()]

# Get PID controller parameters from user input
kp = int(input("Enter Kp value: "))
ki = int(input("Enter Ki value: "))
kd = int(input("Enter Kd value: "))

# --- PID & PLANT TRANSFER FUNCTIONS ---
pid_num = [kd, kp, ki]
pid_den = [1, 0]
pid = ctrl.TransferFunction(pid_num, pid_den)
plant = ctrl.TransferFunction(nums, dens)

# Open-loop and closed-loop systems
open_loop = pid * plant
closed_loop = ctrl.feedback(open_loop)

# --- SIMULATION ---
t = np.linspace(0, 5, 1000)
t_out, y_out = ctrl.step_response(closed_loop, T=t)

# --- TRAINING DATA FOR ANN ---
# Input: time or step input (here we use time, since step input is 1)
X_train = t_out.reshape(-1, 1)  # Feature: time
y_train = y_out                # Target: output of the system

# --- TRAIN ANN MODEL ---
ann = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=5000)
ann.fit(X_train, y_train)

# --- PREDICTION USING ANN ---
y_pred = ann.predict(X_train)

# --- PLOT RESULTS ---
plt.plot(t_out, y_out, label='Actual Closed-loop Response')
plt.plot(t_out, y_pred, 'g--', label='ANN Predicted Response')
plt.plot(t, 1 * np.ones_like(t), 'r--', label='Step Input')
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.title("PID Step Response vs ANN Prediction")
plt.grid(True)
plt.legend()
plt.show()
