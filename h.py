import matplotlib.pyplot as plt
import numpy as np

# Define strategies with coordinates
conservative_coords = [(0, 0), (0, 4), (4, 0), (4, 4), (0, 2), (4, 2), (2, 0), (2, 4), (2, 2)]
moderate_coords = [(0, 0), (0, 4), (4, 0), (4, 4), (2, 0), (2, 4), (0, 2), (4, 2), (2, 2)]
aggressive_coords = [(2, 2), (2, 1), (2, 3), (1, 2), (3, 2), (0, 0), (0, 4), (4, 0), (4, 4)]

# Create a 5x5 grid
fig, ax = plt.subplots()

# Set the limits and ticks of the grid
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_xticks(np.arange(0, 5, 1))
ax.set_yticks(np.arange(0, 5, 1))
ax.grid(True)

# Add coordinates to each cell in the grid
for i in range(5):
    for j in range(5):
        ax.text(i + 0.5, j + 0.5, f'({j+1},{i+1})', va='center', ha='center')

# Plot strategies with different colors
for coord in conservative_coords:
    ax.add_patch(plt.Rectangle((coord[1], coord[0]), 1, 1, fill=True, color='blue', alpha=0.3))
for coord in moderate_coords:
    ax.add_patch(plt.Rectangle((coord[1], coord[0]), 1, 1, fill=True, color='green', alpha=0.3))
for coord in aggressive_coords:
    ax.add_patch(plt.Rectangle((coord[1], coord[0]), 1, 1, fill=True, color='red', alpha=0.3))

# Add a legend
legend_labels = ['Conservative (Blue)', 'Moderate (Green)', 'Aggressive (Red)']
handles = [plt.Rectangle((0, 0), 1, 1, color='blue', alpha=0.3),
           plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.3),
           plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3)]
ax.legend(handles, legend_labels, loc='upper left')

# Set the labels
ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
plt.title('Mines Grid with Strategies')

# Display the grid
plt.gca().invert_yaxis()
plt.show()
