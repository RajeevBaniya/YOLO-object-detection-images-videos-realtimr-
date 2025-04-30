import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('charts/images', exist_ok=True)

# Set the width of the bars
barWidth = 0.15

# Set heights of bars (accuracy percentages)
image_detection = [96.5, 97.2, 95.8, 96.1, 96.4]
video_detection = [95.8, 96.5, 95.2, 95.6, 95.9]
realtime_detection = [95.3, 96.1, 94.7, 94.9, 95.1]

# Set positions of bars on X axis
r1 = np.arange(len(image_detection))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Create figure and axes
plt.figure(figsize=(12, 8))

# Make the plot
bars1 = plt.bar(r1, image_detection, width=barWidth, edgecolor='white', label='Image Detection')
bars2 = plt.bar(r2, video_detection, width=barWidth, edgecolor='white', label='Video Detection')
bars3 = plt.bar(r3, realtime_detection, width=barWidth, edgecolor='white', label='Real-time Detection')

# Removed the labels above bars
# No longer adding values on top of bars

# Add xticks on the middle of the group bars
plt.xlabel('Detection Category', fontweight='bold', fontsize=15)
plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=15)
plt.title('Detection Accuracy Comparison', fontweight='bold', fontsize=18)
plt.xticks([r + barWidth for r in range(len(image_detection))], ['Overall', 'Person', 'Animal', 'Bird', 'Other'])

# Add a legend
plt.legend(loc='lower right', fontsize=12)

# Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Set y-axis to start from 90 for better visualization of differences
plt.ylim(90, 100)

# Add a horizontal line at 95% for reference
plt.axhline(y=95, color='red', linestyle='--', alpha=0.3)

# Save the figure
plt.tight_layout()
plt.savefig('charts/images/accuracy_comparison.png', dpi=300)
plt.close()

print("Accuracy comparison chart saved to 'charts/images/accuracy_comparison.png'") 