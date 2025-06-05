import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import uuid
import json
from datetime import datetime
from utils.utils import initialize, compute_R0
from models.animation_mixin import AnimationMixin

def sample_contacts_with_limit(adj: np.ndarray, infected_idx: np.ndarray, config_params, infection_start_times: np.ndarray, current_step: int, method: tuple = None):
    """
    감염된 노드마다 인접한 노드를 엣지 타입에 따라 샘플링 (각 노드의 감염 지속 시간 고려)
    
    Args:
        adj: 인접 행렬 (adjacency matrix) - 값은 0, 1, 2, 3
        infected_idx: 감염된 노드 인덱스 배열
        config_params: 설정 파라미터 딕셔너리 (N1, N2, N3 포함)
        infection_start_times: 각 노드의 감염 시작 시점 배열
        current_step: 현재 시간 스텝
        method: 방역 대책 (n1, n2, n3) 튜플
    
    Returns:
        contacts: 각 감염 노드별 접촉 노드들의 리스트
    """
    all_contacts = []
    
    for j in infected_idx:
        node_contacts = []
        
        # 이 노드의 감염 지속 시간 계산
        infection_duration = current_step - infection_start_times[j]
        
        # 감염 지속 시간이 limit_starting_step 이상이고 method가 있으면 방역 대책 적용
        if infection_duration >= config_params.get('limit_starting_step', float('inf')) and method is not None:
            n1_mod, n2_mod, n3_mod = method
        else:
            n1_mod, n2_mod, n3_mod = config_params['N1'], config_params['N2'], config_params['N3']
        
        # 노드 j의 이웃들과 엣지 타입 찾기
        neighbors = np.flatnonzero(adj[:, j])  # j로부터 받는 in-neighbors
        
        for neighbor in neighbors:
            edge_type = adj[neighbor, j]  # 엣지 타입 (1, 2, 3)
            
            if edge_type == 1:
                # 타입 1 엣지: N1번 컨택 (방역 대책 적용)
                contact_count = n1_mod
            elif edge_type == 2:
                # 타입 2 엣지: N2번 컨택 (방역 대책 적용)
                contact_count = n2_mod
            elif edge_type == 3:
                # 타입 3 엣지: N3번 컨택 (방역 대책 적용)
                contact_count = n3_mod
            else:
                continue  # 0이면 연결 없음
            
            # 해당 이웃과 contact_count번 컨택
            node_contacts.extend([neighbor] * contact_count)
        
        all_contacts.append(node_contacts)
    
    return all_contacts

def step_with_limit(curr: np.ndarray, adj: np.ndarray, config_params, infection_start_times: np.ndarray, recovery_start_times: np.ndarray, current_step: int, method: tuple = None) -> tuple:
    """
    현재 상태에서 다음 상태로 업데이트 (고정된 기간 후 상태 변화)
    
    Args:
        curr: 현재 상태 벡터 (shape: (N,), dtype: int)
        adj: 인접 행렬 (adjacency matrix)
        config_params: 설정 파라미터 딕셔너리
        infection_start_times: 각 노드의 감염 시작 시점 배열
        recovery_start_times: 각 노드의 회복 시작 시점 배열
        current_step: 현재 시간 스텝
        method: 방역 대책 (n1, n2, n3) 튜플
    
    Returns:
        next: 다음 상태 벡터 (shape: (N,), dtype: int)
        infection_start_times: 업데이트된 감염 시작 시점 배열
        recovery_start_times: 업데이트된 회복 시작 시점 배열
    """
    next = curr.copy()
    infection_start_times = infection_start_times.copy()
    recovery_start_times = recovery_start_times.copy()
    
    # 1. 감염 전파 (S → I): 확률 τ per contact
    infected_idx = np.where(curr == 1)[0]  # 현재 감염된 노드들
    
    if len(infected_idx) > 0:
        # 각 감염 노드가 접촉할 노드들 샘플링 (노드별 감염 지속 시간과 method 고려)
        contacts_list = sample_contacts_with_limit(adj, infected_idx, config_params, infection_start_times, current_step, method)
        
        # 각 감염 노드별로 접촉에 대해 감염 확률 적용
        for i, node_contacts in enumerate(contacts_list):
            for target in node_contacts:
                if curr[target] == 0:  # 대상이 S 상태인 경우
                    if np.random.random() < config_params['tau']:
                        next[target] = 1  # S → I
                        infection_start_times[target] = current_step  # 감염 시작 시점 기록
    
    # 2. 회복 (I → R): 감염된 지 d_tau 스텝 후 회복
    infected_nodes = np.where(curr == 1)[0]
    for node in infected_nodes:
        infection_duration = current_step - infection_start_times[node]
        if infection_duration >= config_params['d_tau']:
            next[node] = 2  # I → R
            recovery_start_times[node] = current_step  # 회복 시작 시점 기록
            infection_start_times[node] = -1  # 감염 시작 시점 초기화
    
    # 3. 면역 소실 (R → S): 회복된 지 d_i 스텝 후 면역 상실
    recovered_nodes = np.where(curr == 2)[0]
    for node in recovered_nodes:
        if recovery_start_times[node] != -1:  # 회복 시작 시점이 기록된 경우
            recovery_duration = current_step - recovery_start_times[node]
            if recovery_duration >= config_params['d_i']:
                next[node] = 0  # R → S
                recovery_start_times[node] = -1  # 회복 시작 시점 초기화
    
    return next, infection_start_times, recovery_start_times

def run_simulation_with_limit(config_params, adj: np.ndarray, init_I: list, method: tuple = None) -> np.ndarray:
    """
    전체 시뮬레이션 실행 (고정된 기간 후 상태 변화)
    
    Args:
        config_params: 설정 파라미터 딕셔너리
        adj: 인접 행렬 (adjacency matrix)
        init_I: 초기 감염 노드 인덱스 리스트
        method: 방역 대책 (n1, n2, n3) 튜플
    
    Returns:
        state: 전체 상태 기록 (shape: (T+1, N), dtype: int)
    """
    # 1. 초기 상태 설정
    state = initialize(config_params, init_I)
    
    # 2. 감염 시작 시점 추적 배열 초기화
    infection_start_times = np.full(config_params['N'], -1, dtype=int)  # -1은 미감염 상태
    for node in init_I:
        infection_start_times[node] = 0  # 초기 감염자들은 0시점에 감염
    
    # 3. 회복 시작 시점 추적 배열 초기화
    recovery_start_times = np.full(config_params['N'], -1, dtype=int)  # -1은 미회복 상태
    
    # 4. 전체 시간에 대한 히스토리 배열 준비 (T+1 스텝)
    history = np.empty((config_params['T'] + 1, config_params['N']), dtype=int)
    history[0] = state  # 초기 상태 저장
    
    # 5. 시간 전진 루프
    for t in range(1, config_params['T'] + 1):
        state, infection_start_times, recovery_start_times = step_with_limit(
            state, adj, config_params, infection_start_times, recovery_start_times, t, method)
        history[t] = state
    
    return history

class EpidemicSimulator(AnimationMixin):
    """SIRS Epidemic Model Simulator with visualization and animation capabilities"""
    
    def __init__(self, config_params=None, adj_matrix=None, init_infected=None, method=None):
        """
        Initialize the epidemic simulator
        
        Args:
            config_params: Configuration parameters dictionary (if None, loads from config_test.json)
            adj_matrix: Pre-generated adjacency matrix (optional)
            init_infected: List of initially infected node indices (if None, uses first 3 nodes)
            method: 방역 대책 (n1, n2, n3) 튜플
        """
        # config_params가 없으면 config_test.json에서 로드
        if config_params is None:
            with open("config_test.json", 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            config_params = config_data.get("defaults", {})
        
        self.config_params = config_params
        self.adj = adj_matrix  # Use provided matrix or will be generated later
        self.init_infected = init_infected if init_infected is not None else list(range(min(3, config_params['N'])))
        self.method = method  # 방역 대책
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
        
        # Run simulation with limit_starting_step and method
        self.state_history = run_simulation_with_limit(self.config_params, self.adj, self.init_infected, self.method)
        
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
    
    def _plot_sir_dynamics(self, custom_filename=None):
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
        
        # method 정보를 title에 포함
        if self.method is not None:
            method_str = f"Method({self.method[0]},{self.method[1]},{self.method[2]})"
            plt.title(f'SIRS Model Simulation - {method_str} (N={self.config_params["N"]}, R0={self.r0:.2f})', fontsize=14)
        else:
            plt.title(f'SIRS Model Simulation (N={self.config_params["N"]}, R0={self.r0:.2f})', fontsize=14)
            
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 파일명 결정
        if custom_filename:
            filename = custom_filename
        elif self.method is not None:
            method_str = f"{self.method[0]}_{self.method[1]}_{self.method[2]}"
            filename = f'sir_dynamics_{method_str}.png'
        else:
            filename = 'sir_dynamics.png'
            
        plt.savefig(os.path.join(self.result_path, filename), dpi=300, bbox_inches='tight')
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
        
        # 매개변수 설정에 따라 표시 방식 결정
        if 'd_tau' in self.config_params and 'd_i' in self.config_params:
            # 새로운 방식: 고정된 기간
            infection_period_info = f"- Infection Period (d_tau): {self.config_params['d_tau']} steps"
            immunity_period_info = f"- Immunity Period (d_i): {self.config_params['d_i']} steps"
            model_type = "Fixed Duration SIRS Model"
        else:
            # 기존 방식: 확률적
            infection_period_info = f"- Recovery Probability (alpha): {self.config_params.get('alpha', 'N/A')}"
            immunity_period_info = f"- Immunity Loss Probability (xi): {self.config_params.get('xi', 'N/A')}"
            model_type = "Stochastic SIRS Model"
        
        info_text = f"""{model_type} Simulation

Simulation Parameters:
- Total Nodes (N): {self.config_params['N']}
- Contact Types (N1, N2, N3): {self.config_params['N1']}, {self.config_params['N2']}, {self.config_params['N3']}
- Infection Probability (tau): {self.config_params['tau']}
{infection_period_info}
{immunity_period_info}
- Time Steps (T): {self.config_params['T']}
- Random Seed: {self.config_params['seed']}
- Limit Starting Step: {self.config_params.get('limit_starting_step', 'N/A')}
- Method Applied: {self.method}

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

    def calculate_custom_metrics(self):
        """
        사용자 정의 지표 계산
        
        Returns:
            dict: 사용자 정의 지표들
        """
        if self.state_history is None:
            return None
        
        # 1. 총 노드 대비 감염된 노드 수 비율 (한번이라도 감염된 적이 있는 노드)
        ever_infected_nodes = np.unique(np.where(self.state_history == 1)[1])
        infection_coverage_ratio = len(ever_infected_nodes) / self.config_params['N']
        
        # 2. 감염 지속 시간 (감염률이 전체의 0.2 밑으로 떨어지는데까지 걸린 시간)
        I_ratios = np.sum(self.state_history == 1, axis=1) / self.config_params['N']
        threshold = 0.2
        
        # 0.2 이상인 time step들 찾기
        above_threshold = I_ratios >= threshold
        
        if np.any(above_threshold):
            # 마지막으로 0.2 이상이었던 time step 찾기
            infection_duration = np.where(above_threshold)[0][-1]  # 마지막 인덱스
        else:
            # 한번도 0.2를 넘지 않았다면 0
            infection_duration = 0
        
        return {
            'infection_coverage_ratio': infection_coverage_ratio,
            'infection_duration': infection_duration,
            'ever_infected_count': len(ever_infected_nodes),
            'method_applied': self.method
        }

    def _create_all_step_network_visualizations(self, method_str=""):
        """Create network visualizations for every 10 steps"""
        G = nx.from_numpy_array(self.adj.astype(int), create_using=nx.DiGraph())
        
        # 노드들이 더 가깝게 모이도록 레이아웃 파라미터 조정
        pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)
        
        color_map = {0: 'lightblue', 1: 'red', 2: 'lightgreen'}
        state_names = {0: 'Susceptible', 1: 'Infected', 2: 'Recovered'}
        
        # 10 간격으로 step points 생성
        key_steps = list(range(0, self.config_params['T'] + 1, 10))
        # 마지막 스텝이 포함되지 않았다면 추가
        if self.config_params['T'] not in key_steps:
            key_steps.append(self.config_params['T'])
        
        print(f"  Creating network visualizations for {len(key_steps)} steps (every 10 steps)...")
        
        for i, step in enumerate(key_steps):
            print(f"    Progress: {i+1}/{len(key_steps)} (step {step})")
                
            plt.figure(figsize=(10, 8))
            
            # Get current state
            current_state = self.state_history[step]
            colors = [color_map[current_state[node]] for node in G.nodes()]
            
            # Draw network
            nx.draw(G, pos, node_color=colors, node_size=100, alpha=0.8,
                    with_labels=False, edge_color='gray', arrows=True,
                    arrowsize=15, arrowstyle='->', width=0.5)
            
            # Add legend
            legend_elements = [plt.scatter([], [], c=color, s=100, label=f'{name} ({np.sum(current_state==i)})')
                              for i, (color, name) in enumerate(zip(color_map.values(), state_names.values()))]
            plt.legend(handles=legend_elements, loc='upper right')
            
            # Title에 method 정보 포함
            if method_str:
                plt.title(f'Network State at Step {step} - Method {method_str}')
            else:
                plt.title(f'Network State at Step {step}')
            plt.axis('off')
            plt.tight_layout()
            
            # 파일명에 method 정보 포함
            if method_str:
                filename = f'network_step_{step:03d}_method_{method_str}.png'
            else:
                filename = f'network_step_{step:03d}.png'
                
            plt.savefig(os.path.join(self.result_path, filename), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Network visualizations saved for {len(key_steps)} steps!")
