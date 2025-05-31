import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx

import numpy as np
import os

class AnimationMixin:
    """
    애니메이션 기능을 제공하는 Mixin 클래스
    EpidemicSimulator에 상속되어 애니메이션 기능을 추가함
    """
    
    def create_network_animation(self, adj, state_history, result_path):
        """Create animated visualization of network state changes"""
        if not self.config.enable_animation:
            print("Animation is disabled in config. Set enable_animation=True to use this feature.")
            return None
            
        print("Creating network animation...")
        G = nx.from_numpy_array(adj.astype(int), create_using=nx.DiGraph())
        pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
        
        # Set up the figure and axis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Color mapping for states
        color_map = {0: 'lightblue', 1: 'red', 2: 'lightgreen'}
        state_names = {0: 'Susceptible', 1: 'Infected', 2: 'Recovered'}
        
        # Initialize network plot
        ax1.set_title('Network State Evolution')
        ax1.axis('off')
        
        # Initialize time series plot
        time_steps = np.arange(self.config.T + 1)
        S_counts = np.sum(state_history == 0, axis=1) / self.config.N
        I_counts = np.sum(state_history == 1, axis=1) / self.config.N
        R_counts = np.sum(state_history == 2, axis=1) / self.config.N
        
        ax2.set_xlim(0, self.config.T)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Population Ratio')
        ax2.set_title('SIR Dynamics Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot full time series as background
        ax2.plot(time_steps, S_counts, color='blue', alpha=0.3, linewidth=1)
        ax2.plot(time_steps, I_counts, color='red', alpha=0.3, linewidth=1)
        ax2.plot(time_steps, R_counts, color='green', alpha=0.3, linewidth=1)
        
        # Initialize dynamic lines
        line_s, = ax2.plot([], [], color='blue', linewidth=2, label='Susceptible')
        line_i, = ax2.plot([], [], color='red', linewidth=2, label='Infected')
        line_r, = ax2.plot([], [], color='green', linewidth=2, label='Recovered')
        ax2.legend()
        
        # Add vertical line for current time
        time_line = ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        
        def animate(frame):
            # Clear previous network plot
            ax1.clear()
            ax1.set_title(f'Network State at Step {frame}')
            ax1.axis('off')
            
            # Get current state
            current_state = state_history[frame]
            colors = [color_map[current_state[node]] for node in G.nodes()]
            
            # Draw network
            nx.draw(G, pos, ax=ax1, node_color=colors, node_size=100, 
                    with_labels=False, edge_color='gray', arrows=True,
                    arrowsize=15, arrowstyle='->', alpha=0.8)
            
            # Add legend for network
            legend_elements = [plt.scatter([], [], c=color, s=100, 
                                         label=f'{name} ({np.sum(current_state==i)})')
                              for i, (color, name) in enumerate(zip(color_map.values(), state_names.values()))]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            # Update time series plot
            current_time = time_steps[:frame+1]
            line_s.set_data(current_time, S_counts[:frame+1])
            line_i.set_data(current_time, I_counts[:frame+1])
            line_r.set_data(current_time, R_counts[:frame+1])
            
            # Update time line
            time_line.set_xdata([frame, frame])
            
            return line_s, line_i, line_r, time_line
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, 
            frames=self.config.T+1, 
            interval=self.config.animation_interval,
            blit=False, 
            repeat=True
        )
        
        # Save animations based on config
        self._save_animations(anim, result_path)
        
        # Show animation if enabled
        if self.config.show_animation:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)
        
        return anim
    
    def _save_animations(self, anim, result_path):
        """Save animation in various formats based on config settings"""
        
        # Save as MP4 if enabled
        if self.config.save_mp4:
            try:
                print("Saving animation as MP4...")
                anim.save(
                    os.path.join(result_path, 'network_evolution.mp4'), 
                    writer='ffmpeg', 
                    fps=self.config.animation_fps, 
                    bitrate=1800
                )
                print("MP4 animation saved successfully!")
            except Exception as e:
                print(f"Could not save MP4 (ffmpeg might not be available): {e}")
        
        # Save as GIF if enabled
        if self.config.save_gif:
            try:
                print("Saving animation as GIF...")
                anim.save(
                    os.path.join(result_path, 'network_evolution.gif'), 
                    writer='pillow', 
                    fps=self.config.animation_fps
                )
                print("GIF animation saved successfully!")
            except Exception as e:
                print(f"Could not save GIF: {e}")
    
    def save_animation_info(self, state_history, result_path):
        """Save detailed animation and simulation information"""
        if not self.config.enable_animation:
            return
            
        # Calculate statistics
        final_state = state_history[-1]
        peak_infected = np.max(np.sum(state_history == 1, axis=1))
        peak_infected_time = np.argmax(np.sum(state_history == 1, axis=1))
        
        total_infected_ever = len(np.unique(np.where(state_history == 1)[1]))
        attack_rate = total_infected_ever / self.config.N
        
        info_text = f"""SIRS Epidemic Model Animation Results

=== Simulation Parameters ===
- Total Nodes (N): {self.config.N}
- Contact Types (N1, N2, N3): {self.config.N1}, {self.config.N2}, {self.config.N3}
- Infection Probability (tau): {self.config.tau}
- Recovery Probability (alpha): {self.config.alpha}
- Immunity Loss Probability (xi): {self.config.xi}
- Time Steps (T): {self.config.T}
- Random Seed: {self.config.seed}

=== Animation Settings ===
- Animation Enabled: {self.config.enable_animation}
- Save MP4: {self.config.save_mp4}
- Save GIF: {self.config.save_gif}
- Show Animation: {self.config.show_animation}
- Animation FPS: {self.config.animation_fps}
- Frame Interval: {self.config.animation_interval}ms

=== Network Properties ===
- Total Contacts per Node: {self.config.N1 + self.config.N2 + self.config.N3}
- Network Connectivity: {(self.config.N1 + self.config.N2 + self.config.N3) / (self.config.N - 1) * 100:.1f}%
- Basic Reproduction Number (R0): {self.r0:.4f}

=== Epidemic Statistics ===
- Peak Infected Count: {peak_infected}
- Peak Infected Time: Step {peak_infected_time}
- Total Ever Infected: {total_infected_ever}
- Attack Rate: {attack_rate:.1%}

=== Final State ===
- Susceptible: {np.sum(final_state == 0)} ({np.sum(final_state == 0)/self.config.N:.1%})
- Infected: {np.sum(final_state == 1)} ({np.sum(final_state == 1)/self.config.N:.1%})
- Recovered: {np.sum(final_state == 2)} ({np.sum(final_state == 2)/self.config.N:.1%})

=== Files Generated ===
"""
        
        if self.config.save_mp4:
            info_text += "- network_evolution.mp4: MP4 animation\n"
        if self.config.save_gif:
            info_text += "- network_evolution.gif: GIF animation\n"
        info_text += "- animation_info.txt: This file\n"
        
        with open(os.path.join(result_path, 'animation_info.txt'), 'w') as f:
            f.write(info_text)
        
        # Save raw animation data
        np.save(os.path.join(result_path, 'animation_state_history.npy'), state_history)
        
        print(f"Animation information saved to: {os.path.join(result_path, 'animation_info.txt')}")
    
    def create_simple_network_gif(self, adj, state_history, result_path, key_frames=None):
        """Create a simple GIF showing key animation frames"""
        if not self.config.enable_animation:
            print("Animation is disabled in config.")
            return None
            
        if key_frames is None:
            # Default key frames: start, 25%, 50%, 75%, end
            key_frames = [0, self.config.T//4, self.config.T//2, 3*self.config.T//4, self.config.T]
            key_frames = [f for f in key_frames if f <= self.config.T]
        
        G = nx.from_numpy_array(adj.astype(int), create_using=nx.DiGraph())
        pos = nx.spring_layout(G, seed=42)
        
        color_map = {0: 'lightblue', 1: 'red', 2: 'lightgreen'}
        state_names = {0: 'Susceptible', 1: 'Infected', 2: 'Recovered'}
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        images = []
        for frame in key_frames:
            ax.clear()
            ax.set_title(f'Network State at Step {frame}')
            ax.axis('off')
            
            current_state = state_history[frame]
            colors = [color_map[current_state[node]] for node in G.nodes()]
            
            nx.draw(G, pos, ax=ax, node_color=colors, node_size=150, 
                    with_labels=False, edge_color='gray', arrows=True,
                    arrowsize=15, arrowstyle='->', alpha=0.8)
            
            # Add legend
            legend_elements = [plt.scatter([], [], c=color, s=100, 
                                         label=f'{name} ({np.sum(current_state==i)})')
                              for i, (color, name) in enumerate(zip(color_map.values(), state_names.values()))]
            ax.legend(handles=legend_elements, loc='upper right')
            
            # Convert plot to image
            fig.canvas.draw()
            
            # Handle different matplotlib versions
            try:
                # Try modern matplotlib method first
                buffer = fig.canvas.buffer_rgba()
                image = np.asarray(buffer)
                # Convert RGBA to RGB
                image = image[:, :, :3]
            except AttributeError:
                try:
                    # Fallback to older method
                    image = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
                    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
                    # Convert ARGB to RGB (skip alpha, reorder)
                    image = image[:, :, 1:4]
                except AttributeError:
                    try:
                        # Another fallback
                        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    except AttributeError:
                        print("Warning: Could not extract image from matplotlib canvas")
                        continue
            
            images.append(image)
        
        plt.close(fig)
        
        # Save as simple GIF using matplotlib
        try:
            from PIL import Image
            pil_images = [Image.fromarray(img) for img in images]
            pil_images[0].save(
                os.path.join(result_path, 'network_keyframes.gif'),
                save_all=True,
                append_images=pil_images[1:],
                duration=1000,  # 1 second per frame
                loop=0
            )
            print("Key frames GIF saved successfully!")
        except ImportError:
            print("PIL not available for simple GIF creation")
        except Exception as e:
            print(f"Could not save key frames GIF: {e}")
        
        return images 