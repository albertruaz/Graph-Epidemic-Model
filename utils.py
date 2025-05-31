from typing import List
import numpy as np
import random
import os
import json


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


def sample_contacts(adj: np.ndarray, infected_idx: np.ndarray, config_params) -> List:
    """
    감염된 노드마다 인접한 노드를 엣지 타입에 따라 샘플링
    
    Args:
        adj: 인접 행렬 (adjacency matrix) - 값은 0, 1, 2, 3
        infected_idx: 감염된 노드 인덱스 배열
        config_params: 설정 파라미터 딕셔너리 (N1, N2, N3 포함)
    
    Returns:
        contacts: 각 감염 노드별 접촉 노드들의 리스트
    """
    all_contacts = []
    
    for j in infected_idx:
        node_contacts = []
        
        # 노드 j의 이웃들과 엣지 타입 찾기
        neighbors = np.flatnonzero(adj[:, j])  # j로부터 받는 in-neighbors
        
        for neighbor in neighbors:
            edge_type = adj[neighbor, j]  # 엣지 타입 (1, 2, 3)
            
            if edge_type == 1:
                # 타입 1 엣지: N1번 컨택
                contact_count = config_params['N1']
            elif edge_type == 2:
                # 타입 2 엣지: N2번 컨택
                contact_count = config_params['N2']
            elif edge_type == 3:
                # 타입 3 엣지: N3번 컨택
                contact_count = config_params['N3']
            else:
                continue  # 0이면 연결 없음
            
            # 해당 이웃과 contact_count번 컨택
            node_contacts.extend([neighbor] * contact_count)
        
        all_contacts.append(node_contacts)
    
    return all_contacts


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
        # 각 감염 노드가 접촉할 노드들 샘플링 (엣지 타입에 따라)
        contacts_list = sample_contacts(adj, infected_idx, config_params)
        
        # 각 감염 노드별로 접촉에 대해 감염 확률 적용
        for i, node_contacts in enumerate(contacts_list):
            for target in node_contacts:
                if curr[target] == 0:  # 대상이 S 상태인 경우
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