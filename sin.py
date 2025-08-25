import numpy as np
import matplotlib.pyplot as plt

# Generate data for sine wave
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='red', linewidth=2)

# Add labels and title
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Wave Plot in Red')

# Add grid for better visualization
plt.grid(True, alpha=0.3)

# Save the plot to a file
plt.savefig('sine_wave_red.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print("Sine wave plot saved as 'sine_wave_red.png'")
