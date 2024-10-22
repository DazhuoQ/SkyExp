import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the number of variables (3 axes, so triangles)
num_vars = 3

# The angle of each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Repeat the first value to close the shape

# Define the 7 sets of values (for 7 triangles)
values_list = [
    [0.6, 0.8, 0.4],
    [0.7, 0.5, 0.9],
    [0.5, 0.7, 0.6],
    [0.8, 0.3, 0.7],
    [0.9, 0.6, 0.5],
    [0.4, 0.9, 0.3],
    [0.7, 0.4, 0.8]
]

# Automatically determine the maximum values for each axis
max_values = [max([values[i] for values in values_list]) for i in range(num_vars)]

# Predefined color palette from earlier (manual color selection)
colors = [
    '#1b9e77', '#66c2a5', '#a6d854',  # Teal/green shades
    '#b8860b', '#e6ab02',             # Darker orange/brown shades
    '#c85a7c', '#8b0000'              # Two rose shades
]

# Use Seaborn style without overriding custom colors
sns.set(style="whitegrid")  # Optional: use Seaborn's styling

# Set up the radar chart
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Plot each triangle in the radar chart with assigned custom colors
for i, values in enumerate(values_list):
    # Normalize the data based on the maximum value for each axis
    normalized_values = [v / max_val for v, max_val in zip(values, max_values)]
    normalized_values += normalized_values[:1]  # To close the shape

    # Plot the radar chart without fill and circles, using our predefined colors
    ax.plot(angles, normalized_values, linewidth=2, linestyle='solid', color=colors[i])

# Remove only the circular grid lines but keep the axis lines
ax.yaxis.grid(False)  # Remove the circular grid lines
ax.xaxis.grid(True)  # Keep the axis lines (spokes)

# Hide the frame (remove circles)
ax.spines['polar'].set_visible(False)

# Set the range of values on the radar chart (0 to 1)
ax.set_ylim(0, 1)

# Show radial ticks with '0' at the center
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])  # Set the radial tick values
ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1'])  # Label the radial ticks

# Set the labels for each axis, including max values
labels = ['Axis 1\n(max: {:.1f})'.format(max_values[0]),
          'Axis 2\n(max: {:.1f})'.format(max_values[1]),
          'Axis 3\n(max: {:.1f})'.format(max_values[2])]

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

# Display the radar chart
plt.show()
