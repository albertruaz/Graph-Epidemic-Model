#!/usr/bin/env python3
"""
학교 반 구조 Matrix 생성 및 저장 도구

사용법:
    python generate_matrix_school.py --name my_school_matrix

기능:
- 학교 반 구조의 adjacency matrix 생성 (30명, 5명씩 6반)
- 반 내부는 완전 연결 (N3), 반간 연결 (N2), 랜덤 연결 (N1)
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
from utils import Config


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


def create_school_class_adjacency_matrix(cfg: Config) -> np.ndarray:
    """
    학교 반 구조의 인접 행렬 생성
    - 30명을 5명씩 6반으로 분할
    - 각 반 내부는 완전 연결 (N3 타입)
    - 1-2반, 3-4반, 5-6반 간 연결 (N2 타입)
    - 랜덤 추가 연결 (N1 타입)
    
    Args:
        cfg: 설정 객체
    
    Returns:
        adj: 인접 행렬 (shape: (N, N), dtype: int, values: 0,1,2,3)
    """
    adj = np.zeros((cfg.N, cfg.N), dtype=int)
    
    # 반 구성 (5명씩 6반)
    students_per_class = 5
    num_classes = cfg.N // students_per_class
    
    classes = []
    for i in range(num_classes):
        start_idx = i * students_per_class
        end_idx = start_idx + students_per_class
        classes.append(list(range(start_idx, end_idx)))
    
    print(f"Created {num_classes} classes with {students_per_class} students each")
    for i, class_students in enumerate(classes):
        print(f"Class {i+1}: {class_students}")
    
    # 1. 각 반 내부 완전 연결 (N3 타입 - 두꺼운 간선)
    for class_students in classes:
        for i, student1 in enumerate(class_students):
            for j, student2 in enumerate(class_students):
                if i != j:  # 자기 자신 제외
                    adj[student2, student1] = 3  # N3 타입
    
    # 2. 반간 연결 (N2 타입 - 중간 간선)
    # 1-2반, 3-4반, 5-6반 연결
    class_pairs = [(0, 1), (2, 3), (4, 5)]  # 0-indexed
    
    for class1_idx, class2_idx in class_pairs:
        if class1_idx < len(classes) and class2_idx < len(classes):
            class1 = classes[class1_idx]
            class2 = classes[class2_idx]
            
            # 각 반에서 N2명씩 선택해서 연결
            connectors1 = np.random.choice(class1, min(cfg.N2, len(class1)), replace=False)
            connectors2 = np.random.choice(class2, min(cfg.N2, len(class2)), replace=False)
            
            # 선택된 학생들끼리 서로 연결
            for student1 in connectors1:
                for student2 in connectors2:
                    adj[student2, student1] = 2  # N2 타입
                    adj[student1, student2] = 2  # N2 타입 (양방향)
            
            print(f"Connected Class {class1_idx+1} and Class {class2_idx+1}: {list(connectors1)} <-> {list(connectors2)}")
    
    # 3. 랜덤 추가 연결 (N1 타입 - 얇은 간선)
    total_random_connections = (cfg.N1 * cfg.N) // 2  # 절반으로 줄임
    
    for _ in range(total_random_connections):
        # 완전히 랜덤하게 두 노드 선택
        student1 = np.random.randint(0, cfg.N)
        student2 = np.random.randint(0, cfg.N)
        
        # 자기 자신이 아니고, 아직 연결되지 않은 경우만
        if student1 != student2 and adj[student2, student1] == 0:
            adj[student2, student1] = 1  # N1 타입
    
    print(f"Added {total_random_connections} random connections (N1 type)")
    
    return adj


def visualize_and_save_matrix(adj: np.ndarray, name: str) -> str:
    """
    학교 반 구조의 Adjacency matrix를 네트워크 그래프로 시각화하고 PNG로 저장
    
    Args:
        adj: 시각화할 adjacency matrix (값: 0,1,2,3)
        name: 저장할 파일명 (확장자 제외)
    
    Returns:
        저장된 이미지 경로
    """
    # NetworkX 그래프 생성
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph())
    
    # 반 구성 정보
    students_per_class = 5
    num_classes = 6
    
    classes = []
    for i in range(num_classes):
        start_idx = i * students_per_class
        end_idx = start_idx + students_per_class
        classes.append(list(range(start_idx, end_idx)))
    
    print(f"Visualizing {num_classes} classes: {classes}")
    
    # 자연스러운 spring layout 사용
    pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)
    
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
    
    # 노드 그리기 (반별 색상)
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightblue',
                          node_size=node_sizes,
                          alpha=0.8,
                          edgecolors='black',
                          linewidths=1.0)
    
    # 간선을 타입별로 그리기 (두께 통일, 색깔로 구분)
    edge_width = 1.5  # 모든 간선 동일한 두께
    edge_colors = {1: 'lightgray', 2: 'gray', 3: 'black'}
    edge_alphas = {1: 0.7, 2: 0.8, 3: 1.0}
    
    for edge_type in [1, 2, 3]:
        edges_of_type = []
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == edge_type:
                    edges_of_type.append((j, i))
        
        if edges_of_type:
            nx.draw_networkx_edges(G, pos,
                                  edgelist=edges_of_type,
                                  edge_color=edge_colors[edge_type],
                                  arrows=True,
                                  arrowsize=6,
                                  arrowstyle='->',
                                  alpha=edge_alphas[edge_type],
                                  width=edge_width)
    
    # 노드 라벨 (모든 학생)
    labels = {i: str(i) for i in range(num_nodes)}
    nx.draw_networkx_labels(G, pos, labels=labels,
                           font_size=7,
                           font_color='white',
                           font_weight='bold')
    
    # 범례 (반별 색상 범례 제거, 간선 타입만)
    legend_elements = [
        plt.Line2D([0], [0], color='lightgray', lw=1.5, label=f'N1 (random, {edges_by_type[1]} edges)'),
        plt.Line2D([0], [0], color='gray', lw=1.5, label=f'N2 (class pairs, {edges_by_type[2]} edges)'),
        plt.Line2D([0], [0], color='black', lw=1.5, label=f'N3 (within class, {edges_by_type[3]} edges)'),
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=10)
    
    plt.title(f'School Class Network: {name}\nNodes: {num_nodes}, Total Edges: {total_edges}, Avg Degree: {avg_degree:.2f}', 
              fontsize=13, fontweight='bold')
    plt.axis('off')
    
    # 우하단 - edge type distribution
    plt.subplot(2, 2, 4)
    types = ['N1\n(Random)', 'N2\n(Class Pairs)', 'N3\n(Within Class)']
    counts = [edges_by_type[1], edges_by_type[2], edges_by_type[3]]
    colors = ['lightgray', 'gray', 'black']
    
    bars = plt.bar(types, counts, color=colors, alpha=0.8, edgecolor='black')
    plt.xlabel('Edge Type', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Edge Type Distribution', fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 이미지 저장
    image_filename = f"./saved_matrix/{name}.png"
    image_path = image_filename
    
    plt.savefig(image_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Network visualization saved to: {image_path}")
    return image_path


def generate_matrix(name: str, N: int = 30, N1: int = 3, N2: int = 2, N3: int = 4):
    """
    학교 반 구조의 매트릭스 생성 및 저장
    
    Args:
        name: 저장할 파일명
        N: 노드 수 (30명 = 5명씩 6반)
        N1: 랜덤 연결 수 (3)
        N2: 반간 연결할 학생 수 (2) 
        N3: 반 내부 연결 수 (4, 5명 중 자기 제외 4명과 연결)
    """
    print(f"=== 학교 반 구조 매트릭스 생성: {name} ===")
    print(f"Parameters: N={N}, N1={N1} (random), N2={N2} (class pairs), N3={N3} (within class)")
    
    # Config 생성
    cfg = Config(
        N=N, N1=N1, N2=N2, N3=N3,
        tau=0.3, alpha=0.1, xi=0.02, T=50,
        seed=42
    )
    
    # 학교 반 구조 adjacency matrix 생성
    adj_matrix = create_school_class_adjacency_matrix(cfg)
    
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
    print(f"N1 (random): {edges_by_type[1]}, N2 (class pairs): {edges_by_type[2]}, N3 (within class): {edges_by_type[3]}")
    
    # JSON으로 저장
    description = f"School class network with {N} students (6 classes of 5), generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
    parser = argparse.ArgumentParser(description='학교 반 구조 네트워크 매트릭스 생성 도구')
    parser.add_argument('--name', type=str, required=True,
                       help='저장할 매트릭스의 이름')
    parser.add_argument('--N', type=int, default=30,
                       help='노드 수 (기본값: 30)')
    parser.add_argument('--N1', type=int, default=2,
                       help='랜덤 연결 수 (기본값: 3)')
    parser.add_argument('--N2', type=int, default=2,
                       help='반간 연결할 학생 수 (기본값: 2)')
    parser.add_argument('--N3', type=int, default=4,
                       help='반 내부 연결 수 (기본값: 4)')
    
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
        print(f"  - 30명이 5명씩 6개 반으로 구성됨")
        print(f"  - N3: 반 내부 완전 연결 (두꺼운 검은선)")
        print(f"  - N2: 1-2반, 3-4반, 5-6반 간 연결 (중간 회색선)")
        print(f"  - N1: 랜덤 추가 연결 (얇은 회색선)")


if __name__ == "__main__":
    main() 