#!/usr/bin/env python3
"""
Matrix 생성 및 저장 도구

사용법:
    python generate_matrix.py --name my_matrix

기능:
- 랜덤 adjacency matrix 생성
- saved_matrix 폴더에 JSON 형식으로 저장
- 네트워크 그래프를 PNG 이미지로 저장 (saved_matrix 상위 폴더에)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from utils.utils import Config


def save_matrix_with_name(adj: np.ndarray, cfg: Config, name: str, description: str = "") -> str:
    """
    Adjacency matrix를 지정된 이름으로 JSON 형식으로 저장
    
    Args:
        adj: 저장할 adjacency matrix
        cfg: Config 객체 (메타데이터용)
        name: 저장할 파일명 (확장자 제외)
        description: 추가 설명
    
    Returns:
        저장된 파일 경로
    """
    # saved_matrix 폴더 생성
    if not os.path.exists("saved_matrix"):
        os.makedirs("saved_matrix")
    
    # 파일명 생성
    filename = f"{name}.json"
    filepath = os.path.join("saved_matrix", filename)
    
    # 파일이 이미 존재하는지 확인
    if os.path.exists(filepath):
        response = input(f"파일 '{filepath}'가 이미 존재합니다. 덮어쓰시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("저장이 취소되었습니다.")
            return None
    
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
            "shape": list(adj.shape),
            "name": name
        }
    }
    
    # JSON 파일로 저장
    with open(filepath, 'w') as f:
        json.dump(matrix_data, f, indent=2)
    
    print(f"Matrix saved to: {filepath}")
    return filepath


def create_typed_random_adjacency_matrix(cfg: Config) -> np.ndarray:
    """
    타입이 구분된 랜덤 인접 행렬 생성 (총 엣지 개수 방식)
    - 0: 연결 없음
    - 1: N1 타입 간선 (총 N1개)
    - 2: N2 타입 간선 (총 N2개)
    - 3: N3 타입 간선 (총 N3개)
    
    Args:
        cfg: 설정 객체
    
    Returns:
        adj: 인접 행렬 (shape: (N, N), dtype: int, values: 0,1,2,3)
    """
    adj = np.zeros((cfg.N, cfg.N), dtype=int)
    
    # 가능한 모든 연결 조합 (자기 자신 제외)
    possible_connections = []
    for i in range(cfg.N):
        for j in range(cfg.N):
            if i != j:  # 자기 자신과의 연결 제외
                possible_connections.append((i, j))
    
    # 총 연결 수가 가능한 연결 수를 초과하지 않도록 제한
    total_edges = cfg.N1 + cfg.N2 + cfg.N3
    max_possible = len(possible_connections)
    
    if total_edges > max_possible:
        print(f"Warning: 요청된 총 엣지 수 ({total_edges})가 최대 가능 수 ({max_possible})를 초과합니다.")
        # 비율에 따라 조정
        ratio = max_possible / total_edges
        cfg.N1 = int(cfg.N1 * ratio)
        cfg.N2 = int(cfg.N2 * ratio) 
        cfg.N3 = int(cfg.N3 * ratio)
        total_edges = cfg.N1 + cfg.N2 + cfg.N3
        print(f"조정된 엣지 수: N1={cfg.N1}, N2={cfg.N2}, N3={cfg.N3}")
    
    # 랜덤하게 연결 선택
    selected_connections = np.random.choice(len(possible_connections), 
                                          total_edges, 
                                          replace=False)
    
    # 선택된 연결들을 타입별로 배정
    edge_types = ([1] * cfg.N1 + [2] * cfg.N2 + [3] * cfg.N3)
    np.random.shuffle(edge_types)  # 타입 순서 섞기
    
    # 인접 행렬에 연결 추가
    for idx, edge_type in zip(selected_connections, edge_types):
        i, j = possible_connections[idx]
        adj[i, j] = edge_type
    
    print(f"Created random network with N1={cfg.N1}, N2={cfg.N2}, N3={cfg.N3} edges")
    
    return adj


def visualize_and_save_matrix(adj: np.ndarray, name: str) -> str:
    """
    타입이 구분된 Adjacency matrix를 네트워크 그래프로 시각화하고 PNG로 저장
    
    Args:
        adj: 시각화할 adjacency matrix (값: 0,1,2,3)
        name: 저장할 파일명 (확장자 제외)
    
    Returns:
        저장된 이미지 경로
    """
    # NetworkX 그래프 생성
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
    
    # 자연스러운 spring layout 사용
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100)
    
    # 노드 크기를 degree에 따라 조정
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees.values() else 1
    node_sizes = [200 + (degrees[node] / max_degree) * 200 for node in G.nodes()]
    
    # 그래프 정보 계산
    num_nodes = G.number_of_nodes()
    edges_by_type = {
        1: np.sum(adj == 1),
        2: np.sum(adj == 2), 
        3: np.sum(adj == 3)
    }
    total_edges = sum(edges_by_type.values())
    avg_degree = total_edges * 2 / num_nodes if num_nodes > 0 else 0
    
    # 플롯 생성
    plt.figure(figsize=(14, 10))
    
    # 메인 플롯 - 네트워크 그래프
    plt.subplot(2, 2, (1, 3))
    
    # 노드 그리기 (모든 노드 같은 색상)
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',  # 모든 노드 같은 색
                          node_size=node_sizes,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1.0)
    
    # 간선을 타입별로 그리기 (두께 통일, 색깔로 구분)
    edge_width = 1.5  # 모든 간선 동일한 두께
    edge_colors = {1: 'lightgray', 2: 'gray', 3: 'black'}
    edge_alphas = {1: 0.7, 2: 0.8, 3: 1.0}
    
    for edge_type in [1, 2, 3]:
        # 해당 타입의 간선들만 찾기
        edges_of_type = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == edge_type:
                    edges_of_type.append((j, i))  # j -> i 방향
        
        if edges_of_type:
            nx.draw_networkx_edges(G, pos,
                                  edgelist=edges_of_type,
                                  edge_color=edge_colors[edge_type],
                                  arrows=True,
                                  arrowsize=6,
                                  arrowstyle='->',
                                  alpha=edge_alphas[edge_type],
                                  width=edge_width)
    
    # 노드 라벨 그리기 (노드 수가 적을 때만)
    if num_nodes <= 30:
        nx.draw_networkx_labels(G, pos, 
                               font_size=7,
                               font_color='white',
                               font_weight='bold')
    
    # 범례 (간선 타입만)
    legend_elements = [
        plt.Line2D([0], [0], color='lightgray', lw=1.5, label=f'N1 ({edges_by_type[1]} edges)'),
        plt.Line2D([0], [0], color='gray', lw=1.5, label=f'N2 ({edges_by_type[2]} edges)'),
        plt.Line2D([0], [0], color='black', lw=1.5, label=f'N3 ({edges_by_type[3]} edges)')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    
    plt.title(f'Random Network: {name}\nNodes: {num_nodes}, Total Edges: {total_edges}, Avg Degree: {avg_degree:.2f}', 
              fontsize=13, fontweight='bold')
    plt.axis('off')
    
    # 우하단 - edge type distribution
    plt.subplot(2, 2, 4)
    types = ['N1', 'N2', 'N3']
    counts = [edges_by_type[1], edges_by_type[2], edges_by_type[3]]
    colors = ['lightgray', 'gray', 'black']
    
    bars = plt.bar(types, counts, color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('Edge Type', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Edge Type Distribution', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 숫자 표시
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 전체 레이아웃 조정
    plt.tight_layout()
    
    # 이미지 저장
    image_filename = f"./saved_matrix/{name}.png"
    image_path = image_filename
    
    plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Network visualization saved to: {image_path}")
    return image_path


def generate_matrix(name: str, N: int = 30, N1: int = 40, N2: int = 30, N3: int = 20):
    """
    랜덤 매트릭스 생성 및 저장 (총 엣지 개수 방식)
    
    Args:
        name: 저장할 파일명
        N: 노드 수
        N1, N2, N3: 각 타입별 총 엣지 개수
    """
    print(f"=== 랜덤 매트릭스 생성: {name} ===")
    print(f"Parameters: N={N}, N1={N1} (total edges), N2={N2} (total edges), N3={N3} (total edges)")
    
    # Config 생성 (랜덤 매트릭스용)
    cfg = Config(
        N=N, N1=N1, N2=N2, N3=N3,
        tau=0.3, alpha=0.1, xi=0.02, T=50,
        seed=42
    )
    
    # 랜덤 adjacency matrix 생성
    adj_matrix = create_typed_random_adjacency_matrix(cfg)
    
    # 매트릭스 정보 출력
    total_edges = np.sum(adj_matrix > 0)
    edges_by_type = {
        1: np.sum(adj_matrix == 1),
        2: np.sum(adj_matrix == 2), 
        3: np.sum(adj_matrix == 3)
    }
    density = total_edges / (N * (N - 1))  # 자기 자신 제외
    print(f"Matrix density: {density:.3f}")
    print(f"Total edges: {total_edges}")
    print(f"N1 edges: {edges_by_type[1]}, N2 edges: {edges_by_type[2]}, N3 edges: {edges_by_type[3]}")
    
    # JSON으로 저장
    description = f"Random network with {N} nodes, {total_edges} edges, generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    json_path = save_matrix_with_name(adj_matrix, cfg, name, description)
    
    if json_path:
        # PNG 이미지로 시각화 저장
        image_path = visualize_and_save_matrix(adj_matrix, name)
        
        print(f"\n=== 생성 완료 ===")
        print(f"Matrix JSON: {json_path}")
        print(f"Graph PNG: {image_path}")
        
        return json_path, image_path
    else:
        return None, None


def main():
    parser = argparse.ArgumentParser(description='랜덤 네트워크 매트릭스 생성 도구 (총 엣지 개수 방식)')
    parser.add_argument('--name', type=str, required=True,
                       help='저장할 매트릭스의 이름')
    parser.add_argument('--N', type=int, default=90,
                       help='노드 수 (기본값: 30)')
    parser.add_argument('--N1', type=int, default=120,
                       help='N1 타입 총 엣지 개수 (기본값: 20)')
    parser.add_argument('--N2', type=int, default=90,
                       help='N2 타입 총 엣지 개수 (기본값: 15)')
    parser.add_argument('--N3', type=int, default=60,
                       help='N3 타입 총 엣지 개수 (기본값: 10)')
    
    args = parser.parse_args()
    
    # 매트릭스 생성
    json_path, image_path = generate_matrix(
        name=args.name,
        N=args.N,
        N1=args.N1,
        N2=args.N2,
        N3=args.N3
    )
    
    if json_path and image_path:
        print(f"\n사용법:")
        print(f"  config.json에서 'saved_matrix_path': '{json_path}'로 설정하여 사용")
        print(f"  그래프 모양은 {image_path}에서 확인 가능")
        print(f"\n특징:")
        print(f"  - 총 {args.N1 + args.N2 + args.N3}개의 엣지가 랜덤하게 배치됨")
        print(f"  - N1: {args.N1}개 (얇은 회색선)")
        print(f"  - N2: {args.N2}개 (중간 회색선)")
        print(f"  - N3: {args.N3}개 (두꺼운 검은선)")


if __name__ == "__main__":
    main() 