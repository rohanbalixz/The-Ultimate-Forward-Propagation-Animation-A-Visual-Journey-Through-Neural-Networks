import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import matplotlib.cm as cm

input_neurons = 3
hidden_neurons = 3
output_neurons = 1

np.random.seed(42)
W1 = np.random.randn(hidden_neurons, input_neurons)
b1 = np.random.randn(hidden_neurons, 1)
W2 = np.random.randn(output_neurons, hidden_neurons)
b2 = np.random.randn(output_neurons, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0.5], [0.8], [0.2]])

z1_final = np.dot(W1, X) + b1
a1_final = sigmoid(z1_final)
z2_final = np.dot(W2, a1_final) + b2
a2_final = sigmoid(z2_final)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1, 7)
ax.set_ylim(-3, 3)
ax.axis('off')
ax.set_facecolor('whitesmoke')

title_text = ax.text(3, 2.5, "", fontsize=18, ha='center', va='center', fontweight='bold', color='darkblue')

def neuron_positions(n, x_center, y_center=0, spacing=1.5):
    positions = []
    offset = spacing * (n - 1) / 2
    for i in range(n):
        positions.append((x_center, y_center + spacing * i - offset))
    return positions

input_pos = neuron_positions(input_neurons, 0)
hidden_pos = neuron_positions(hidden_neurons, 3)
output_pos = neuron_positions(output_neurons, 6)

neurons = {'input': [], 'hidden': [], 'output': []}
texts = {'input': [], 'hidden': [], 'output': []}

for layer, positions in zip(['input', 'hidden', 'output'], [input_pos, hidden_pos, output_pos]):
    for (x, y) in positions:
        circle = patches.Circle((x, y), radius=0.3, edgecolor='black', facecolor='lightgray', lw=2, zorder=2)
        ax.add_patch(circle)
        neurons[layer].append(circle)
        text = ax.text(x, y, "", fontsize=12, ha='center', va='center', zorder=3, fontweight='bold')
        texts[layer].append(text)

arrowprops = dict(arrowstyle="->", color="darkgray", lw=2, connectionstyle="arc3,rad=0.3")
arrows = []
def draw_arrows(start_positions, end_positions):
    for (x0, y0) in start_positions:
        for (x1, y1) in end_positions:
            arrow = ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=arrowprops)
            arrows.append(arrow)
draw_arrows(input_pos, hidden_pos)
draw_arrows(hidden_pos, output_pos)

steps_count = 6
frames_per_step = 30
total_frames = steps_count * frames_per_step

def lerp_color(color1, color2, t):
    """Linearly interpolate between two RGBA colors."""
    return tuple((1 - t) * c1 + t * c2 for c1, c2 in zip(color1, color2))

def get_hidden_color(val):
    return cm.viridis(val)
def get_output_color(val):
    return cm.plasma(val)

def animate(frame):
    step = frame // frames_per_step
    t = (frame % frames_per_step) / frames_per_step

    if step == 0:
        title_text.set_text("Step 0: Input Values")
        for i, text in enumerate(texts['input']):
            text.set_text(f'{X[i, 0]:.2f}')
            text.set_alpha(t)
            neurons['input'][i].set_facecolor(lerp_color((0.8, 0.9, 1, 1), (0.6, 0.8, 1, 1), t))
    elif step == 1:
        title_text.set_text("Step 1: Hidden Layer Weighted Sum (z1)")
        for i, text in enumerate(texts['hidden']):
            current_val = t * z1_final[i, 0]
            text.set_text(f'{current_val:.2f}')
            text.set_alpha(t)
            neurons['hidden'][i].set_facecolor(lerp_color((0.9, 0.9, 0.9, 1), (1, 1, 0.8, 1), t))
    elif step == 2:
        title_text.set_text("Step 2: Hidden Layer Activation (a1)")
        for i, text in enumerate(texts['hidden']):
            current_val = t * a1_final[i, 0]
            text.set_text(f'{current_val:.2f}')
            text.set_alpha(t)
            final_color = get_hidden_color(a1_final[i, 0])
            neurons['hidden'][i].set_facecolor(lerp_color((1, 1, 1, 1), final_color, t))
    elif step == 3:
        title_text.set_text("Step 3: Output Layer Weighted Sum (z2)")
        for i, text in enumerate(texts['output']):
            current_val = t * z2_final[i, 0]
            text.set_text(f'{current_val:.2f}')
            text.set_alpha(t)
            neurons['output'][i].set_facecolor(lerp_color((1, 0.9, 0.9, 1), (1, 0.8, 0.8, 1), t))
    elif step == 4:
        title_text.set_text("Step 4: Output Layer Activation (a2)")
        for i, text in enumerate(texts['output']):
            current_val = t * a2_final[i, 0]
            text.set_text(f'{current_val:.2f}')
            text.set_alpha(t)
            final_color = get_output_color(a2_final[i, 0])
            neurons['output'][i].set_facecolor(lerp_color((1, 1, 1, 1), final_color, t))
    elif step == 5:
        title_text.set_text("Forward Propagation Complete!")
        for layer in texts:
            for text in texts[layer]:
                text.set_alpha(1)
    return []

ani = animation.FuncAnimation(fig, animate, frames=total_frames, interval=50, repeat=False)
plt.show()
