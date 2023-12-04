import numpy as np
import matplotlib.pyplot as plt

# Define the AR values
AR = np.linspace(1.01, 10, 400)  # Assuming AR values from just above 1 to 10
e = np.sqrt(1 - 1/AR**2)

# Plot
plt.figure(figsize=(10,6))
plt.plot(AR, e, '-b', label='e vs AR')
plt.title('Relationship between Eccentricity (e) and Aspect Ratio (AR)')
plt.xlabel('Aspect Ratio (AR)')
plt.ylabel('Eccentricity (e)')
plt.grid(True)
plt.legend()
plt.show()