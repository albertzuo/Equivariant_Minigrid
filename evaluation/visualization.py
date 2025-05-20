import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import os

def create_animation(frames, save_path=None):
    """
    Create animation from frames.
    Args:
        frames: List of frames to animate
        save_path: Optional path to save the animation as MP4
    """
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        im.set_array(frames[i])
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=200, blit=True)
    
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the animation
        anim.save(save_path, writer='ffmpeg', fps=5)
    
    plt.close(fig)
    return HTML(anim.to_jshtml())

def plot_parameter_comparison(custom_params, sb3_params):
    plt.figure(figsize=(8, 5))
    models = ['Custom PPO', 'Stable Baselines3 PPO']
    params = [custom_params, sb3_params]
    colors = ['skyblue', 'lightgreen']
    plt.bar(models, params, color=colors)
    plt.title('Model Parameter Count Comparison')
    plt.ylabel('Number of Parameters')
    plt.yscale('log')  # Log scale for better visualization if there's a big difference
    plt.tight_layout()
    plt.show()
