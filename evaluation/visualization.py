import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def create_animation(frames):
    """Create animation from frames."""
    
    fig = plt.figure(figsize=(5, 5))
    im = plt.imshow(frames[0])
    plt.axis('off')
    
    def animate(i):
        im.set_array(frames[i])
        return [im]
    
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=200, blit=True)
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
