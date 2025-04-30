import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create directory if it doesn't exist
os.makedirs('charts/images', exist_ok=True)

# Set up the figure with white background
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 16), facecolor='white')
ax.set_facecolor('white')

# Set axis limits
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)

# Define modern color palette
colors = {
    'start_end': '#4285F4',      # Google Blue
    'process': '#34A853',        # Google Green
    'decision': '#FBBC05',       # Google Yellow
    'data': '#EA4335',           # Google Red
    'arrow': '#5F6368',          # Google Grey
    'text': '#202124',           # Google Black
    'white': '#FFFFFF',          # White
}

# Function to create a rounded rectangle
def create_rounded_box(x, y, width, height, label, color, alpha=1.0):
    # Create rectangle with rounded corners
    box = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=patches.BoxStyle("Round", pad=0.6, radius=0.15),
        facecolor=color,
        edgecolor='none',
        alpha=alpha,
        linewidth=0,
        zorder=2
    )
    ax.add_patch(box)
    
    # Add text with good contrast
    text_color = 'white' if color != colors['white'] else colors['text']
    ax.text(x + width/2, y + height/2, label,
           ha='center', va='center',
           fontsize=12, fontweight='bold',
           color=text_color,
           zorder=3)
    
    return (x, y, width, height)

# Function to create a diamond
def create_diamond(x, y, width, height, label, color, alpha=1.0):
    diamond = patches.RegularPolygon((x + width/2, y + height/2), 4, 
                                     radius=width/1.414, 
                                     orientation=np.pi/4,
                                     facecolor=color,
                                     edgecolor='none',
                                     linewidth=0, 
                                     alpha=alpha,
                                     zorder=2)
    ax.add_patch(diamond)
    ax.text(x + width/2, y + height/2, label,
           ha='center', va='center',
           fontsize=11, fontweight='bold',
           color='white' if color != colors['white'] else colors['text'],
           zorder=3)
    return (x, y, width, height)

# Function to create a parallelogram
def create_parallelogram(x, y, width, height, label, color, alpha=1.0):
    # Create a parallelogram shape using a polygon patch
    p = patches.Polygon([[x+0.3, y], [x+width, y], [x+width-0.3, y+height], [x, y+height]],
                        facecolor=color,
                        edgecolor='none',
                        linewidth=0, 
                        alpha=alpha,
                        zorder=2)
    ax.add_patch(p)
    ax.text(x + width/2, y + height/2, label,
           ha='center', va='center',
           fontsize=11, fontweight='bold',
           color='white' if color != colors['white'] else colors['text'],
           zorder=3)
    return (x, y, width, height)

# Function to create an oval
def create_oval(x, y, width, height, label, color, alpha=1.0):
    ellipse = patches.Ellipse((x + width/2, y + height/2), width, height,
                             facecolor=color,
                             edgecolor='none',
                             linewidth=0, 
                             alpha=alpha,
                             zorder=2)
    ax.add_patch(ellipse)
    ax.text(x + width/2, y + height/2, label,
           ha='center', va='center',
           fontsize=12, fontweight='bold',
           color='white' if color != colors['white'] else colors['text'],
           zorder=3)
    return (x, y, width, height)

# Function to draw a curved arrow between components
def draw_arrow(start, end, label=None, style='arc3,rad=0.1', bidirectional=False):
    # Calculate center points
    if isinstance(start, tuple) and len(start) == 4:  # (x, y, width, height)
        start_x = start[0] + start[2]/2
        start_y = start[1]  # Bottom of box
    else:
        start_x, start_y = start
    
    if isinstance(end, tuple) and len(end) == 4:  # (x, y, width, height)
        end_x = end[0] + end[2]/2
        end_y = end[1] + end[3]  # Top of box
    else:
        end_x, end_y = end
    
    # Create elegant arrow
    arrow = patches.FancyArrowPatch(
        (start_x, start_y), (end_x, end_y),
        arrowstyle='-|>' if not bidirectional else '<|-|>',
        connectionstyle=style,
        linewidth=2,
        color=colors['arrow'],
        zorder=1
    )
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        # Position label with good readability based on the curve orientation
        rad = float(style.split('=')[1])
        if rad > 0:  # Curve to the right
            text_x = (start_x + end_x) / 2 + abs(rad) * 1.5
            text_y = (start_y + end_y) / 2
            ha = 'left'
        elif rad < 0:  # Curve to the left
            text_x = (start_x + end_x) / 2 - abs(rad) * 1.5
            text_y = (start_y + end_y) / 2
            ha = 'right'
        else:  # Straight line
            text_x = (start_x + end_x) / 2 + 0.2
            text_y = (start_y + end_y) / 2
            ha = 'left'
        
        ax.text(text_x, text_y, label,
               ha=ha, va='center',
               fontsize=10, 
               bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.8),
               zorder=3)

# Title
ax.text(5, 15.5, "SmartSentry System Workflow", fontsize=18, fontweight='bold', ha='center', color=colors['text'])
ax.text(5, 15.1, "Intelligent Object Detection Pipeline", fontsize=14, ha='center', color='#666666', style='italic')

# Box dimensions
box_width = 2.5
box_height = 0.8

# Create workflow components
start = create_oval(3.75, 14, box_width, box_height, "Start", colors['start_end'])

input_source = create_parallelogram(3.75, 12.5, box_width, box_height, "Input Source", colors['data'])

preprocessing = create_rounded_box(3.75, 11, box_width, box_height, "Preprocessing", colors['process'])

# Detection stage
detection = create_rounded_box(3.75, 9.5, box_width, box_height, "YOLO Detection", colors['process'])

# Decision diamonds
confidence_check = create_diamond(3.75, 8, box_width, box_height, "Confidence\n> Threshold?", colors['decision'])

# Classification paths
category_mapping = create_rounded_box(3.75, 6.5, box_width, box_height, "Category Mapping", colors['process'])
skip_detection = create_rounded_box(1, 8, box_width, box_height, "Skip Detection", colors['process'])

# Additional processing
nms = create_rounded_box(3.75, 5, box_width, box_height, "Non-Max Suppression", colors['process'])

# Output options
visualization = create_rounded_box(1, 3.5, box_width, box_height, "Visualization", colors['process'])
metrics = create_rounded_box(3.75, 3.5, box_width, box_height, "Metrics Calculation", colors['process'])
alert = create_rounded_box(6.5, 3.5, box_width, box_height, "Alert System", colors['process'])

# End
end = create_oval(3.75, 2, box_width, box_height, "End", colors['start_end'])

# Connect with arrows
draw_arrow(start, input_source)
draw_arrow(input_source, preprocessing, "Images/Videos/Camera Feed")
draw_arrow(preprocessing, detection, "Normalized Input")
draw_arrow(detection, confidence_check, "Object Candidates")

# Decision paths
draw_arrow(confidence_check, category_mapping, "Yes", "arc3,rad=0")
draw_arrow(confidence_check, skip_detection, "No", "arc3,rad=-0.3")

# Continue main flow
draw_arrow(category_mapping, nms)

# Output paths
draw_arrow(nms, visualization, "Filtered Detections", "arc3,rad=-0.2")
draw_arrow(nms, metrics)
draw_arrow(nms, alert, "If Alert Required", "arc3,rad=0.2")

# Connect all to end
draw_arrow(visualization, end, "", "arc3,rad=-0.2")
draw_arrow(metrics, end)
draw_arrow(alert, end, "", "arc3,rad=0.2")
draw_arrow(skip_detection, end, "", "arc3,rad=-0.3")

# Add annotation boxes
annotations = [
    (6.5, 11, "• Resize to 416×416\n• Normalize pixel values\n• Channel reordering"),
    (6.5, 9.5, "• YOLOv3 with Darknet-53\n• Multi-scale detection"),
    (6.5, 6.5, "• Map COCO classes to:\n  Person, Animal, Bird, Other"),
    (6.5, 5, "• Eliminate overlapping boxes\n• IoU threshold: 0.3")
]

for x, y, text in annotations:
    # Create an annotation box with a subtle background
    ax.text(x, y, text, fontsize=10, va="center", ha="left", color=colors['text'],
           bbox=dict(facecolor='#F8F9FA', edgecolor='#DADCE0', boxstyle='round,pad=0.5', alpha=0.9))

# Add a legend at the bottom
legend_elements = [
    patches.Patch(facecolor=colors['start_end'], label='Start/End'),
    patches.Patch(facecolor=colors['process'], label='Process'),
    patches.Patch(facecolor=colors['decision'], label='Decision'),
    patches.Patch(facecolor=colors['data'], label='Input/Output Data')
]

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
         ncol=4, frameon=False, fontsize=10)

# Turn off axis
ax.set_axis_off()

# Save the figure with higher quality
plt.tight_layout()
plt.savefig('charts/images/system_workflow.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Modern colorful workflow diagram saved to 'charts/images/system_workflow.png'") 