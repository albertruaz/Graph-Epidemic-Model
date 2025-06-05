import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import uuid
from datetime import datetime
from utils.utils import initialize, run_simulation, compute_R0
from models.animation_mixin import AnimationMixin

class EpidemicSimulator(AnimationMixin):
    """SIRS Epidemic Model Simulator with visualization and animation capabilities"""
    
    def __init__(self, config_params, adj_matrix=None, init_infected=None):
        """
        Initialize the epidemic simulator
        
        Args:
            config_params: Configuration parameters dictionary
            adj_matrix: Pre-generated adjacency matrix (optional)
            init_infected: List of initially infected node indices (if None, uses first 3 nodes)
        """
        self.config_params = config_params
        self.adj = adj_matrix  # Use provided matrix or will be generated later
        self.init_infected = init_infected if init_infected is not None else list(range(min(3, config_params['N'])))
        self.state_history = None
        self.r0 = None
        self.result_path = None
        
    def create_result_folder(self, folder_prefix="result"):
        """Create unique result folder with timestamp and UUID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        folder_name = f"{folder_prefix}_{timestamp}_{unique_id}"
        
        if not os.path.exists("results"):
            os.makedirs("results")
        
        result_path = os.path.join("results", folder_name)
        os.makedirs(result_path)
        
        return result_path
    
    def run_simulation(self, save_results=True, create_plots=True, folder_prefix="result"):
        """
        Run the epidemic simulation
        
        Args:
            save_results: Whether to save results to files
            create_plots: Whether to create and save plots
            folder_prefix: Prefix for result folder name
            
        Returns:
            dict: Simulation results including state_history, r0, statistics
        """
        # Create result folder if saving results
        if save_results:
            self.result_path = self.create_result_folder(folder_prefix)
            print(f"Results will be saved to: {self.result_path}")
        
        
        # Run simulation
        self.state_history = run_simulation(self.config_params, self.adj, self.init_infected)
        
        # Calculate R0
        self.r0 = compute_R0(self.config_params)
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        if save_results:
            # Save simulation info
            self._save_simulation_info(stats)
            
            if create_plots:
                # Plot SIR dynamics
                self._plot_sir_dynamics()
                
                # Create network visualizations for key time steps
                self._create_network_visualizations()
            
            # Handle animation features if enabled
            if self.config_params['enable_animation']:
                print("Animation is enabled. Creating animations...")
                # Create full animation
                self.create_network_animation(self.adj, self.state_history, self.result_path)
                
                # Create simple key frames GIF
                self.create_simple_network_gif(self.adj, self.state_history, self.result_path)
                
                # Save animation info
                self.save_animation_info(self.state_history, self.result_path)
            else:
                print("Animation is disabled. To enable animations, set enable_animation=True in Config.")
        
        return {
            'state_history': self.state_history,
            'adjacency_matrix': self.adj,
            'r0': self.r0,
            'statistics': stats,
            'result_path': self.result_path
        }
    
    def _calculate_statistics(self):
        """Calculate epidemic statistics"""
        if self.state_history is None:
            return None
            
        # Basic counts over time
        S_counts = np.sum(self.state_history == 0, axis=1)
        I_counts = np.sum(self.state_history == 1, axis=1) 
        R_counts = np.sum(self.state_history == 2, axis=1)
        
        # Peak statistics
        peak_infected = np.max(I_counts)
        peak_infected_time = np.argmax(I_counts)
        peak_infected_ratio = peak_infected / self.config_params['N']
        
        # Final state
        final_state = self.state_history[-1]
        final_susceptible = np.sum(final_state == 0)
        final_infected = np.sum(final_state == 1)
        final_recovered = np.sum(final_state == 2)
        
        # Attack rate (total ever infected)
        total_infected_ever = len(np.unique(np.where(self.state_history == 1)[1]))
        attack_rate = total_infected_ever / self.config_params['N']
        
        # Epidemic duration (time with >1% infected)
        threshold = 0.01 * self.config_params['N']
        above_threshold = I_counts > threshold
        epidemic_duration = np.sum(above_threshold) if np.any(above_threshold) else 0
        
        return {
            'S_counts': S_counts,
            'I_counts': I_counts,
            'R_counts': R_counts,
            'peak_infected': peak_infected,
            'peak_infected_time': peak_infected_time,
            'peak_infected_ratio': peak_infected_ratio,
            'final_susceptible': final_susceptible,
            'final_infected': final_infected,
            'final_recovered': final_recovered,
            'total_infected_ever': total_infected_ever,
            'attack_rate': attack_rate,
            'epidemic_duration': epidemic_duration
        }
    
    def _plot_sir_dynamics(self):
        """Plot SIR dynamics over time"""
        S_ratios = (self.state_history == 0).sum(axis=1) / self.config_params['N']
        I_ratios = (self.state_history == 1).sum(axis=1) / self.config_params['N']
        R_ratios = (self.state_history == 2).sum(axis=1) / self.config_params['N']
        
        plt.figure(figsize=(12, 8))
        time_steps = np.arange(self.config_params['T'] + 1)
        plt.plot(time_steps, S_ratios, label='Susceptible', color='blue', linewidth=2)
        plt.plot(time_steps, I_ratios, label='Infected', color='red', linewidth=2)
        plt.plot(time_steps, R_ratios, label='Recovered', color='green', linewidth=2)
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Population Ratio', fontsize=12)
        plt.title(f'SIRS Model Simulation (N={self.config_params["N"]}, R0={self.r0:.2f})', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.result_path, 'sir_dynamics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_network_visualizations(self):
        """Create network visualizations for key time steps"""
        G = nx.from_numpy_array(self.adj.astype(int), create_using=nx.DiGraph())
        
        # 노드들이 더 가깝게 모이도록 레이아웃 파라미터 조정
        pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)  # k값을 줄여서 노드들을 더 가깝게
        
        # 10 간격으로 key time points 생성
        key_steps = list(range(0, self.config_params['T'] + 1, 10))
        # 마지막 스텝이 포함되지 않았다면 추가
        if self.config_params['T'] not in key_steps:
            key_steps.append(self.config_params['T'])
        
        color_map = {0: 'lightblue', 1: 'red', 2: 'lightgreen'}
        state_names = {0: 'Susceptible', 1: 'Infected', 2: 'Recovered'}
        
        for step in key_steps:
            plt.figure(figsize=(10, 8))
            
            # Get current state
            current_state = self.state_history[step]
            colors = [color_map[current_state[node]] for node in G.nodes()]
            
            # Draw network - 노드 크기와 간격 조정
            nx.draw(G, pos, node_color=colors, node_size=100, alpha=0.8,
                    with_labels=False, edge_color='gray', arrows=True,
                    arrowsize=15, arrowstyle='->', width=0.5)
            
            # Add legend
            legend_elements = [plt.scatter([], [], c=color, s=100, label=f'{name} ({np.sum(current_state==i)})')
                              for i, (color, name) in enumerate(zip(color_map.values(), state_names.values()))]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title(f'Network State at Step {step}')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_path, f'network_step_{step:03d}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def _save_simulation_info(self, stats):
        """Save simulation parameters and results"""
        info_text = f"""SIRS Epidemic Model Simulation

Simulation Parameters:
- Total Nodes (N): {self.config_params['N']}
- Contact Types (N1, N2, N3): {self.config_params['N1']}, {self.config_params['N2']}, {self.config_params['N3']}
- Infection Probability (tau): {self.config_params['tau']}
- Recovery Probability (alpha): {self.config_params['alpha']}
- Immunity Loss Probability (xi): {self.config_params['xi']}
- Time Steps (T): {self.config_params['T']}
- Random Seed: {self.config_params['seed']}

Calculated Metrics:
- Basic Reproduction Number (R0): {self.r0:.4f}
- Total Contacts per Node: {self.config_params['N1'] + self.config_params['N2'] + self.config_params['N3']}
- Network Connectivity: {(self.config_params['N1'] + self.config_params['N2'] + self.config_params['N3']) / (self.config_params['N'] - 1) * 100:.1f}%

Epidemic Statistics:
- Peak Infected Count: {stats['peak_infected']}
- Peak Infected Time: Step {stats['peak_infected_time']}
- Peak Infected Ratio: {stats['peak_infected_ratio']:.1%}
- Total Ever Infected: {stats['total_infected_ever']}
- Attack Rate: {stats['attack_rate']:.1%}
- Epidemic Duration: {stats['epidemic_duration']} steps

Final State:
- Susceptible: {stats['final_susceptible']} ({stats['final_susceptible']/self.config_params['N']:.1%})
- Infected: {stats['final_infected']} ({stats['final_infected']/self.config_params['N']:.1%})
- Recovered: {stats['final_recovered']} ({stats['final_recovered']/self.config_params['N']:.1%})

Results saved in: {self.result_path}
"""
        
        with open(os.path.join(self.result_path, 'simulation_info.txt'), 'w') as f:
            f.write(info_text)
