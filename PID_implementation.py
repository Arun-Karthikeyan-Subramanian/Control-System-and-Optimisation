from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Get plant transfer function from user input
nums = [int(x) for x in input("Enter numerator: ").split()]
dens = [int(x) for x in input("Enter denominator: ").split()]

# Get PID controller parameters from user input
kp = int(input("Enter Kp value:"))
ki = int(input("Enter Ki value:"))
kd = int(input("Enter Kd value:"))

# PID transfer function
pid_num = [kd, kp, ki]  # This represents Kd * s^2 + Kp * s + Ki
pid_den = [1, 0]        # This represents 's'

# Create the PID controller transfer function
pid = ctrl.TransferFunction(pid_num, pid_den)

# Create the plant transfer function
plant = ctrl.TransferFunction(nums, dens)

# Open loop transfer function (PID * plant)
open_loop = pid * plant

# Closed loop transfer function with feedback
closed_loop = ctrl.feedback(open_loop)

# Time vector for simulation
t = np.linspace(0, 5, 1000)

# Get the step response of the closed loop system
t_out, y_out = ctrl.step_response(closed_loop, T=t)

# Plot the results
plt.plot(t_out, y_out, label='Closed-loop Response')
plt.plot(t, 1 * np.ones_like(t), 'r--', label='Step Input')
plt.xlabel("Time [s]")
plt.ylabel("Output")
plt.title("Step Response of Closed-Loop System with PID")
plt.grid(True)
plt.legend()
plt.show()


