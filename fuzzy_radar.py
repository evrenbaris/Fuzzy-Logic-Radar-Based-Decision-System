import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Simulate radar data
num_samples = 100
distance = np.random.uniform(0, 500, num_samples)  # Distance in meters
speed = np.random.uniform(0, 100, num_samples)  # Speed in m/s
angle = np.random.uniform(0, 180, num_samples)  # Angle in degrees

# Create a DataFrame to hold radar data
radar_data = pd.DataFrame({'Distance': distance, 'Speed': speed, 'Angle': angle})

# Define fuzzy variables
distance = ctrl.Antecedent(np.arange(0, 501, 1), 'Distance')
speed = ctrl.Antecedent(np.arange(0, 101, 1), 'Speed')
danger = ctrl.Consequent(np.arange(0, 101, 1), 'Danger')

# Define membership functions for Distance
distance['near'] = fuzz.trapmf(distance.universe, [0, 0, 50, 100])
distance['medium'] = fuzz.trimf(distance.universe, [50, 150, 300])
distance['far'] = fuzz.trapmf(distance.universe, [200, 400, 500, 500])

# Define membership functions for Speed
speed['slow'] = fuzz.trapmf(speed.universe, [0, 0, 20, 40])
speed['medium'] = fuzz.trimf(speed.universe, [20, 50, 80])
speed['fast'] = fuzz.trapmf(speed.universe, [60, 80, 100, 100])

# Define membership functions for Danger
danger['low'] = fuzz.trapmf(danger.universe, [0, 0, 20, 40])
danger['medium'] = fuzz.trimf(danger.universe, [20, 50, 80])
danger['high'] = fuzz.trapmf(danger.universe, [60, 80, 100, 100])

# Visualize membership functions
distance.view()
speed.view()
danger.view()

# Define fuzzy rules
rule1 = ctrl.Rule(distance['near'] & speed['fast'], danger['high'])
rule2 = ctrl.Rule(distance['near'] & speed['medium'], danger['medium'])
rule3 = ctrl.Rule(distance['far'] & speed['slow'], danger['low'])
rule4 = ctrl.Rule(distance['medium'] & speed['medium'], danger['medium'])
rule5 = ctrl.Rule(distance['far'] & speed['fast'], danger['medium'])

# Create a control system and simulation
danger_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
danger_simulation = ctrl.ControlSystemSimulation(danger_control)

# Test the system with sample input
sample_distance = 50  # Example distance in meters
sample_speed = 70  # Example speed in m/s

danger_simulation.input['Distance'] = sample_distance
danger_simulation.input['Speed'] = sample_speed
danger_simulation.compute()

# Output the result
print(f"Danger Level: {danger_simulation.output['Danger']:.2f}")

# Visualize the result
danger.view(sim=danger_simulation)
