from typing import List, Optional
import numpy as np
import random


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


def initialize(cfg: Config, init_I: List[int]) -> np.ndarray:
    """
    초기 상태 벡터 생성
    
    Args:
        cfg: 설정 객체
        init_I: 초기 감염 노드 인덱스 리스트
    
    Returns:
        state: 초기 상태 벡터 (shape: (N,), dtype: int)
    """
    # 상태값: 0=S(Susceptible), 1=I(Infected), 2=R(Recovered)
    # PDF 모델과 동일하게 벡터를 0으로 초기화 (모든 노드가 S 상태)
    state = np.zeros(cfg.N, dtype=int)
    
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


def step(curr: np.ndarray, adj: np.ndarray, cfg: Config) -> np.ndarray:
    """
    현재 상태에서 다음 상태로 업데이트
    
    Args:
        curr: 현재 상태 벡터 (shape: (N,), dtype: int)
        adj: 인접 행렬 (adjacency matrix)
        cfg: 설정 객체
    
    Returns:
        next: 다음 상태 벡터 (shape: (N,), dtype: int)
    """
    # PDF 식 (16.1)에 따른 상태 전이 구현
    next = curr.copy()
    
    # 1. 감염 전파 (S → I): 확률 τ per contact
    infected_idx = np.where(curr == 1)[0]  # 현재 감염된 노드들
    
    if len(infected_idx) > 0:
        # 각 감염 노드가 접촉할 노드들 샘플링
        total_contacts = cfg.N1 + cfg.N2 + cfg.N3
        contacts = sample_contacts(adj, infected_idx, total_contacts)
        
        # 각 접촉에 대해 감염 확률 적용
        for i in range(len(infected_idx)):
            for k in range(total_contacts):
                target = contacts[i, k]
                if target >= 0 and curr[target] == 0:  # 대상이 유효하고 S 상태인 경우
                    if np.random.random() < cfg.tau:
                        next[target] = 1  # S → I
    
    # 2. 회복 (I → R): 확률 α per node per step
    infected_nodes = np.where(curr == 1)[0]
    for node in infected_nodes:
        if np.random.random() < cfg.alpha:
            next[node] = 2  # I → R
    
    # 3. 면역 소실 (R → S): 확률 ξ per node per step
    recovered_nodes = np.where(curr == 2)[0]
    for node in recovered_nodes:
        if np.random.random() < cfg.xi:
            next[node] = 0  # R → S
    
    return next


def run_simulation(cfg: Config, adj: np.ndarray, init_I: List[int]) -> np.ndarray:
    """
    전체 시뮬레이션 실행
    
    Args:
        cfg: 설정 객체
        adj: 인접 행렬 (adjacency matrix)
        init_I: 초기 감염 노드 인덱스 리스트
    
    Returns:
        state: 전체 상태 기록 (shape: (T+1, N), dtype: int)
    """
    # PDF MATLAB 코드의 "for n=1:T" 블록과 동일한 구조
    # 1. 초기 상태 설정
    state = initialize(cfg, init_I)
    
    # 2. 전체 시간에 대한 히스토리 배열 준비 (T+1 스텝)
    history = np.empty((cfg.T + 1, cfg.N), dtype=int)
    history[0] = state  # 초기 상태 저장
    
    # 3. 시간 전진 루프
    for t in range(1, cfg.T + 1):
        state = step(state, adj, cfg)
        history[t] = state
    
    return history


def compute_R0(cfg: Config) -> float:
    """
    기본 재생산 지수(R0) 계산
    
    Args:
        cfg: 설정 객체
    
    Returns:
        R0: 기본 재생산 지수 값
    """
    # PDF 모델 식 (16.1)에 따른 R0 계산
    # Stochastic IBM에서 "1 타임스텝 평균 접촉 수"가 N1+N2+N3
    # β = τ × (N1+N2+N3)/N, α = cfg.alpha
    # R0 = β*N / α = τ*(N1+N2+N3)/α
    
    total_contacts = cfg.N1 + cfg.N2 + cfg.N3
    R0 = cfg.tau * total_contacts / cfg.alpha
    
    return R0


def create_adjacency_matrix(cfg: Config) -> np.ndarray:
    """
    설정에 따라 인접 행렬 생성
    
    Args:
        cfg: 설정 객체
    
    Returns:
        adj: 인접 행렬 (shape: (N, N), dtype: bool)
    """
    # PDF 코드 예시의 `randi`로 노드별 out-degree 생성을 파이썬화
    # 열 j가 'j → i' 가능한 엣지를 나타내도록 설정
    
    adj = np.zeros((cfg.N, cfg.N), dtype=bool)
    
    for j in range(cfg.N):
        # 각 노드 j의 out-degree는 N1+N2+N3
        outdeg = cfg.N1 + cfg.N2 + cfg.N3
        
        # 자기 자신을 제외한 노드들 중에서 무작위로 outdeg개 선택
        available_targets = np.delete(np.arange(cfg.N), j)
        
        if len(available_targets) >= outdeg:
            targets = np.random.choice(available_targets, outdeg, replace=False)
        else:
            # 가능한 모든 노드를 선택 (outdeg가 N-1보다 큰 경우)
            targets = available_targets
        
        # adj[targets, j] = True는 "j → targets" 엣지를 의미
        adj[targets, j] = True
    
    return adj 