from typing import List, Optional
import numpy as np
import random
import os
import json
import uuid
from datetime import datetime


class Config:
    """
    시뮬레이션 설정을 저장하는 클래스
    """
    def __init__(
        self,
        N: int,
        N1: int,
        N2: int,
        N3: int,
        tau: float,
        alpha: float,
        xi: float,
        T: int,
        seed: Optional[int] = None,
        enable_animation: bool = False,
        save_mp4: bool = True,
        save_gif: bool = True,
        show_animation: bool = True,
        animation_fps: int = 5,
        animation_interval: int = 200
    ):
        self.N = N          # 전체 노드 수
        self.N1 = N1        # 첫 번째 종류의 contact 수
        self.N2 = N2        # 두 번째 종류의 contact 수
        self.N3 = N3        # 세 번째 종류의 contact 수
        self.tau = tau      # 감염 확률
        self.alpha = alpha  # 회복 확률
        self.xi = xi        # 면역 소실 확률
        self.T = T          # 최대 시뮬레이션 스텝 수
        self.seed = seed    # 랜덤 시드
        
        # 애니메이션 관련 설정
        self.enable_animation = enable_animation    # 애니메이션 활성화 여부
        self.save_mp4 = save_mp4                   # MP4 저장 여부
        self.save_gif = save_gif                   # GIF 저장 여부
        self.show_animation = show_animation        # 애니메이션 화면 출력 여부
        self.animation_fps = animation_fps          # 애니메이션 FPS
        self.animation_interval = animation_interval # 애니메이션 프레임 간격 (ms)
        
        # 시드가 제공되었다면 설정
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)


def initialize(config_params, init_I: List[int]) -> np.ndarray:
    """
    초기 상태 벡터 생성
    
    Args:
        config_params: 설정 파라미터 딕셔너리
        init_I: 초기 감염 노드 인덱스 리스트
    
    Returns:
        state: 초기 상태 벡터 (shape: (N,), dtype: int)
    """
    # 상태값: 0=S(Susceptible), 1=I(Infected), 2=R(Recovered)
    # PDF 모델과 동일하게 벡터를 0으로 초기화 (모든 노드가 S 상태)
    state = np.zeros(config_params['N'], dtype=int)
    
    # 초기 감염자들을 I 상태(1)로 설정
    state[init_I] = 1
    
    return state


def sample_contacts(adj: np.ndarray, infected_idx: np.ndarray, k: int) -> np.ndarray:
    """
    감염된 노드마다 인접한 노드 중 k개를 샘플링
    
    Args:
        adj: 인접 행렬 (adjacency matrix)
        infected_idx: 감염된 노드 인덱스 배열
        k: 각 감염 노드마다 샘플링할 인접 노드 수
    
    Returns:
        contacts: 샘플링된 접촉 노드들 (shape: (len(infected_idx), k), dtype: int)
    """
    # PDF 모델의 수식에 따라 접촉은 그래프의 방향을 따름
    # adj[:, j]는 노드 j로부터 받는 in-neighbors를 나타냄
    num_infected = len(infected_idx)
    contacts = np.full((num_infected, k), -1, dtype=int)  # -1로 패딩 초기화
    
    for i, j in enumerate(infected_idx):
        # 노드 j의 이웃 노드들 찾기 (PDF의 find(C[:,j]==1)과 동일)
        neighbors = np.flatnonzero(adj[:, j])
        
        if len(neighbors) > 0:
            # 이웃 중에서 최대 k개까지 중복 없이 샘플링
            num_to_sample = min(k, len(neighbors))
            sampled = np.random.choice(neighbors, num_to_sample, replace=False)
            contacts[i, :num_to_sample] = sampled
    
    return contacts


def step(curr: np.ndarray, adj: np.ndarray, config_params) -> np.ndarray:
    """
    현재 상태에서 다음 상태로 업데이트
    
    Args:
        curr: 현재 상태 벡터 (shape: (N,), dtype: int)
        adj: 인접 행렬 (adjacency matrix)
        config_params: 설정 파라미터 딕셔너리
    
    Returns:
        next: 다음 상태 벡터 (shape: (N,), dtype: int)
    """
    # PDF 식 (16.1)에 따른 상태 전이 구현
    next = curr.copy()
    
    # 1. 감염 전파 (S → I): 확률 τ per contact
    infected_idx = np.where(curr == 1)[0]  # 현재 감염된 노드들
    
    if len(infected_idx) > 0:
        # 각 감염 노드가 접촉할 노드들 샘플링
        total_contacts = config_params['N1'] + config_params['N2'] + config_params['N3']
        contacts = sample_contacts(adj, infected_idx, total_contacts)
        
        # 각 접촉에 대해 감염 확률 적용
        for i in range(len(infected_idx)):
            for k in range(total_contacts):
                target = contacts[i, k]
                if target >= 0 and curr[target] == 0:  # 대상이 유효하고 S 상태인 경우
                    if np.random.random() < config_params['tau']:
                        next[target] = 1  # S → I
    
    # 2. 회복 (I → R): 확률 α per node per step
    infected_nodes = np.where(curr == 1)[0]
    for node in infected_nodes:
        if np.random.random() < config_params['alpha']:
            next[node] = 2  # I → R
    
    # 3. 면역 소실 (R → S): 확률 ξ per node per step
    recovered_nodes = np.where(curr == 2)[0]
    for node in recovered_nodes:
        if np.random.random() < config_params['xi']:
            next[node] = 0  # R → S
    
    return next


def run_simulation(config_params, adj: np.ndarray, init_I: List[int]) -> np.ndarray:
    """
    전체 시뮬레이션 실행
    
    Args:
        config_params: 설정 파라미터 딕셔너리
        adj: 인접 행렬 (adjacency matrix)
        init_I: 초기 감염 노드 인덱스 리스트
    
    Returns:
        state: 전체 상태 기록 (shape: (T+1, N), dtype: int)
    """
    # PDF MATLAB 코드의 "for n=1:T" 블록과 동일한 구조
    # 1. 초기 상태 설정
    state = initialize(config_params, init_I)
    
    # 2. 전체 시간에 대한 히스토리 배열 준비 (T+1 스텝)
    history = np.empty((config_params['T'] + 1, config_params['N']), dtype=int)
    history[0] = state  # 초기 상태 저장
    
    # 3. 시간 전진 루프
    for t in range(1, config_params['T'] + 1):
        state = step(state, adj, config_params)
        history[t] = state
    
    return history


def compute_R0(config_params) -> float:
    """
    기본 재생산 지수(R0) 계산
    
    Args:
        config_params: 설정 파라미터 딕셔너리
    
    Returns:
        R0: 기본 재생산 지수 값
    """
    # PDF 모델 식 (16.1)에 따른 R0 계산
    # Stochastic IBM에서 "1 타임스텝 평균 접촉 수"가 N1+N2+N3
    # β = τ × (N1+N2+N3)/N, α = cfg.alpha
    # R0 = β*N / α = τ*(N1+N2+N3)/α
    
    total_contacts = config_params['N1'] + config_params['N2'] + config_params['N3']
    R0 = config_params['tau'] * total_contacts / config_params['alpha']
    
    return R0


def create_result_folder():
    """결과 저장용 고유 폴더 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    folder_name = f"result_{timestamp}_{unique_id}"
    
    if not os.path.exists("results"):
        os.makedirs("results")
    
    result_path = os.path.join("results", folder_name)
    os.makedirs(result_path)
    
    return result_path


def load_saved_adjacency_matrix(matrix_path: str) -> np.ndarray:
    """저장된 adjacency matrix 불러오기 (타입이 있는 새로운 형식과 기존 bool 형식 모두 지원)"""
    try:
        with open(matrix_path, 'r') as f:
            data = json.load(f)
        
        # JSON에서 numpy 배열로 변환
        adj_raw = np.array(data['adjacency_matrix'])
        
        # 새로운 형식 (int 타입, 값: 0,1,2,3)인지 확인
        if adj_raw.dtype == int and np.max(adj_raw) > 1:
            # 새로운 형식: 타입이 있는 매트릭스
            print(f"Loading typed adjacency matrix from: {matrix_path}")
            
            # 타입별 엣지 개수 출력
            edges_by_type = {
                1: np.sum(adj_raw == 1),
                2: np.sum(adj_raw == 2), 
                3: np.sum(adj_raw == 3)
            }
            total_edges = sum(edges_by_type.values())
            N = adj_raw.shape[0]
            print(f"Loaded matrix: N={N}, Total edges={total_edges}")
            print(f"N1={edges_by_type[1]}, N2={edges_by_type[2]}, N3={edges_by_type[3]}")
            
            return adj_raw
        
        else:
            # 기존 형식: bool 타입 매트릭스
            print(f"Loading boolean adjacency matrix from: {matrix_path}")
            adj = adj_raw.astype(bool)
            
            return adj
        
    except FileNotFoundError:
        raise FileNotFoundError(f"저장된 matrix 파일을 찾을 수 없습니다: {matrix_path}")
    except json.JSONDecodeError:
        raise ValueError(f"잘못된 JSON 형식입니다: {matrix_path}")
    except Exception as e:
        raise ValueError(f"Matrix 로딩 중 오류 발생: {e}")


def save_matrix(adj: np.ndarray, cfg: Config, description: str = "") -> str:
    """
    Adjacency matrix를 JSON 형식으로 저장
    
    Args:
        adj: 저장할 adjacency matrix
        cfg: Config 객체 (메타데이터용)
        description: 추가 설명
    
    Returns:
        저장된 파일 경로
    """
    # saved_matrix 폴더 생성
    if not os.path.exists("saved_matrix"):
        os.makedirs("saved_matrix")
    
    # 고유 파일명 생성 (create_result_folder 방식)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"matrix_{timestamp}_{unique_id}.json"
    filepath = os.path.join("saved_matrix", filename)
    
    # 저장할 데이터 구성
    matrix_data = {
        "adjacency_matrix": adj.astype(int).tolist(),  # JSON 호환을 위해 int로 변환
        "metadata": {
            "N": cfg.N,
            "N1": cfg.N1,
            "N2": cfg.N2,
            "N3": cfg.N3,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "shape": list(adj.shape)
        }
    }
    
    # JSON 파일로 저장
    with open(filepath, 'w') as f:
        json.dump(matrix_data, f, indent=2)
    
    print(f"Matrix saved to: {filepath}")
    return filepath


def load_config_from_json(config_file="config.json", simulation_name=None):
    """
    JSON 파일에서 설정을 읽어서 Config 객체와 매트릭스 경로를 생성
    
    Args:
        config_file: JSON 설정 파일 경로
        simulation_name: 사용할 시뮬레이션 설정 이름 (없으면 defaults 사용)
    
    Returns:
        tuple: (Config 객체, 매트릭스 파일 경로)
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        if simulation_name and simulation_name in config_data.get("simulations", {}):
            params = config_data["simulations"][simulation_name]
            print(f"Loaded configuration for: {simulation_name}")
        else:
            params = config_data.get("defaults", {})
            print("Loaded default configuration")
        
        # "save_matrix" 설정 처리 - 파일명만 지정하면 saved_matrix 폴더에서 찾기
        matrix_path = None
        if "save_matrix" in params:
            matrix_name = params["save_matrix"]
            if matrix_name:
                # .json 확장자가 없으면 추가
                if not matrix_name.endswith('.json'):
                    matrix_name += '.json'
                
                # saved_matrix 폴더에서 파일 경로 생성
                matrix_path = os.path.join("saved_matrix", matrix_name)
                
                # 파일이 존재하는지 확인
                if os.path.exists(matrix_path):
                    print(f"Matrix file found: {matrix_path}")
                else:
                    print(f"Warning: Matrix file not found: {matrix_path}")
                    matrix_path = None
            
            # "save_matrix" 키는 Config 클래스에 없으므로 제거
            del params["save_matrix"]
        
        # Config 클래스에 없는 키들 제거
        config_only_keys = ["init_infected_count", "init_infected_method", "init_infected_indices"]
        for key in config_only_keys:
            if key in params:
                del params[key]
        
        # None 값들을 실제 None으로 변환
        for key, value in params.items():
            if value is None:
                params[key] = None
        
        return Config(**params), matrix_path
        
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default settings.")
        return Config(N=50, N1=3, N2=2, N3=1, tau=0.3, alpha=0.1, xi=0.02, T=50), None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return Config(N=50, N1=3, N2=2, N3=1, tau=0.3, alpha=0.1, xi=0.02, T=50), None
    except Exception as e:
        print(f"Error loading config: {e}")
        return Config(N=50, N1=3, N2=2, N3=1, tau=0.3, alpha=0.1, xi=0.02, T=50), None


def update_config_in_json(config_file, simulation_name, updates):
    """
    JSON 설정 파일의 특정 시뮬레이션 설정을 업데이트
    
    Args:
        config_file: JSON 설정 파일 경로
        simulation_name: 업데이트할 시뮬레이션 설정 이름
        updates: 업데이트할 설정 dict
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        if "simulations" not in config_data:
            config_data["simulations"] = {}
        
        if simulation_name not in config_data["simulations"]:
            config_data["simulations"][simulation_name] = {}
        
        config_data["simulations"][simulation_name].update(updates)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
            
        print(f"Updated configuration for {simulation_name}")
        
    except Exception as e:
        print(f"Error updating config: {e}")


def generate_init_infected(config_params):
    """
    설정에 따라 초기 감염자 리스트 생성
    
    Args:
        config_params: 설정 파라미터 dict
    
    Returns:
        list: 초기 감염자 인덱스 리스트
    """
    method = config_params.get("init_infected_method", "sequential")
    count = config_params.get("init_infected_count", 3)
    N = config_params.get("N", 50)
    
    if method == "sequential":
        # 순차적으로 처음부터
        return list(range(min(count, N)))
    
    elif method == "random":
        # 랜덤하게 선택
        return list(np.random.choice(N, min(count, N), replace=False))
    
    elif method == "specific":
        # 특정 인덱스들
        indices = config_params.get("init_infected_indices", [0])
        return [idx for idx in indices if idx < N]
    
    else:
        # 기본값: 처음 count개
        return list(range(min(count, N))) 