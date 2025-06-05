import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns
import re
import os
import sys
from pathlib import Path

def parse_simulation_results(file_path):
    """
    시뮬레이션 결과 파일을 파싱하여 DataFrame으로 변환
    
    Args:
        file_path: 결과 파일 경로
    
    Returns:
        pandas.DataFrame: 파싱된 데이터
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 데이터 라인 찾기 (헤더 이후부터)
    data_start = False
    
    for line in lines:
        line = line.strip()
        
        # 헤더 라인 확인
        if "Method(N1,N2,N3)" in line:
            data_start = True
            continue
        
        # 구분선 건너뛰기
        if "---" in line or not line:
            continue
            
        # 데이터 라인 파싱
        if data_start and line.startswith('('):
            # (2, 4, 8)	0.5778	28	52	2.9400 형태 파싱
            parts = line.split('\t')
            if len(parts) >= 5:
                # Method 파싱: (2, 4, 8) -> [2, 4, 8]
                method_str = parts[0]
                method_match = re.findall(r'\d+', method_str)
                if len(method_match) == 3:
                    n1, n2, n3 = map(int, method_match)
                    
                    # 나머지 값들 파싱
                    infection_coverage_ratio = float(parts[1])
                    infection_duration = int(parts[2])
                    ever_infected_count = int(parts[3])
                    r0 = float(parts[4])
                    
                    # Duration/Coverage 비율 계산
                    duration_coverage_ratio = infection_duration / infection_coverage_ratio if infection_coverage_ratio > 0 else 0
                    
                    data.append({
                        'N1': n1,
                        'N2': n2,
                        'N3': n3,
                        'Infection_Coverage_Ratio': infection_coverage_ratio,
                        'Infection_Duration': infection_duration,
                        'Duration_Coverage_Ratio': duration_coverage_ratio,
                        'Ever_Infected_Count': ever_infected_count,
                        'R0': r0
                    })
    
    return pd.DataFrame(data)

def create_heatmaps_by_n1(df, output_dir, metrics=['Infection_Coverage_Ratio', 'Infection_Duration', 'Duration_Coverage_Ratio']):
    """
    N1값별로 (N2, N3) 히트맵 생성
    
    Args:
        df: 파싱된 데이터프레임
        output_dir: 출력 디렉토리
        metrics: 히트맵을 생성할 지표들
    """
    # N1의 고유값들 찾기
    n1_values = sorted(df['N1'].unique())
    n2_values = sorted(df['N2'].unique())
    n3_values = sorted(df['N3'].unique())
    
    print(f"Found N1 values: {n1_values}")
    print(f"Found N2 values: {n2_values}")
    print(f"Found N3 values: {n3_values}")
    
    for metric in metrics:
        print(f"\nCreating heatmaps for {metric}...")
        
        # 각 N1값에 대해 히트맵 생성
        fig, axes = plt.subplots(1, len(n1_values), figsize=(6*len(n1_values), 5))
        if len(n1_values) == 1:
            axes = [axes]
        
        for i, n1_val in enumerate(n1_values):
            # N1값에 해당하는 데이터 필터링
            n1_data = df[df['N1'] == n1_val]
            
            # N2, N3를 인덱스로 하는 피벗 테이블 생성
            pivot_table = n1_data.pivot_table(
                values=metric, 
                index='N3',  # y축 (위에서 아래로)
                columns='N2',  # x축 (왼쪽에서 오른쪽으로)
                fill_value=np.nan
            )
            
            # 인덱스 정렬 (N3는 내림차순으로 정렬하여 위쪽이 큰 값)
            pivot_table = pivot_table.sort_index(ascending=False)
            
            # 히트맵 생성
            fmt_str = '.3f' if 'Ratio' in metric else '.0f'
            if metric == 'Duration_Coverage_Ratio':
                fmt_str = '.1f'
            
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt=fmt_str,
                cmap='YlOrRd',
                ax=axes[i],
                cbar_kws={'label': metric}
            )
            
            axes[i].set_title(f'N1 = {n1_val}')
            axes[i].set_xlabel('N2 (Contact Type 2)')
            axes[i].set_ylabel('N3 (Contact Type 3)')
        
        plt.tight_layout()
        
        # 파일 저장
        output_file = os.path.join(output_dir, f'heatmap_{metric}_by_N1.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def create_individual_heatmaps(df, output_dir, metrics=['Infection_Coverage_Ratio', 'Infection_Duration', 'Duration_Coverage_Ratio']):
    """
    각 N1, metric 조합에 대해 개별 히트맵 생성
    
    Args:
        df: 파싱된 데이터프레임
        output_dir: 출력 디렉토리
        metrics: 히트맵을 생성할 지표들
    """
    n1_values = sorted(df['N1'].unique())
    
    for metric in metrics:
        for n1_val in n1_values:
            print(f"Creating individual heatmap for N1={n1_val}, {metric}...")
            
            # N1값에 해당하는 데이터 필터링
            n1_data = df[df['N1'] == n1_val]
            
            # 피벗 테이블 생성
            pivot_table = n1_data.pivot_table(
                values=metric, 
                index='N3',
                columns='N2',
                fill_value=np.nan
            )
            
            # 인덱스 정렬
            pivot_table = pivot_table.sort_index(ascending=False)
            
            # 개별 히트맵 생성
            plt.figure(figsize=(10, 8))
            
            # 포맷 설정
            fmt_str = '.3f' if 'Ratio' in metric else '.0f'
            if metric == 'Duration_Coverage_Ratio':
                fmt_str = '.1f'
            
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt=fmt_str,
                cmap='YlOrRd',
                cbar_kws={'label': metric}
            )
            
            plt.title(f'{metric} Heatmap (N1 = {n1_val})', fontsize=16)
            plt.xlabel('N2 (Contact Type 2)', fontsize=12)
            plt.ylabel('N3 (Contact Type 3)', fontsize=12)
            plt.tight_layout()
            
            # 파일 저장
            output_file = os.path.join(output_dir, f'heatmap_N1_{n1_val}_{metric}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()

def generate_summary_stats(df, output_dir):
    """
    요약 통계 생성 및 저장
    
    Args:
        df: 파싱된 데이터프레임
        output_dir: 출력 디렉토리
    """
    summary_file = os.path.join(output_dir, 'summary_statistics.txt')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== Simulation Results Summary Statistics ===\n\n")
        
        f.write(f"Total number of simulations: {len(df)}\n")
        f.write(f"N1 values: {sorted(df['N1'].unique())}\n")
        f.write(f"N2 values: {sorted(df['N2'].unique())}\n")
        f.write(f"N3 values: {sorted(df['N3'].unique())}\n\n")
        
        f.write("=== Infection Coverage Ratio Statistics ===\n")
        f.write(f"Mean: {df['Infection_Coverage_Ratio'].mean():.4f}\n")
        f.write(f"Std:  {df['Infection_Coverage_Ratio'].std():.4f}\n")
        f.write(f"Min:  {df['Infection_Coverage_Ratio'].min():.4f}\n")
        f.write(f"Max:  {df['Infection_Coverage_Ratio'].max():.4f}\n\n")
        
        f.write("=== Infection Duration Statistics ===\n")
        f.write(f"Mean: {df['Infection_Duration'].mean():.2f}\n")
        f.write(f"Std:  {df['Infection_Duration'].std():.2f}\n")
        f.write(f"Min:  {df['Infection_Duration'].min()}\n")
        f.write(f"Max:  {df['Infection_Duration'].max()}\n\n")
        
        f.write("=== Duration/Coverage Ratio Statistics ===\n")
        f.write(f"Mean: {df['Duration_Coverage_Ratio'].mean():.2f}\n")
        f.write(f"Std:  {df['Duration_Coverage_Ratio'].std():.2f}\n")
        f.write(f"Min:  {df['Duration_Coverage_Ratio'].min():.2f}\n")
        f.write(f"Max:  {df['Duration_Coverage_Ratio'].max():.2f}\n\n")
        
        # N1별 통계
        f.write("=== Statistics by N1 ===\n")
        for n1_val in sorted(df['N1'].unique()):
            n1_data = df[df['N1'] == n1_val]
            f.write(f"\nN1 = {n1_val}:\n")
            f.write(f"  Infection Coverage Ratio: {n1_data['Infection_Coverage_Ratio'].mean():.4f} ± {n1_data['Infection_Coverage_Ratio'].std():.4f}\n")
            f.write(f"  Infection Duration: {n1_data['Infection_Duration'].mean():.2f} ± {n1_data['Infection_Duration'].std():.2f}\n")
            f.write(f"  Duration/Coverage Ratio: {n1_data['Duration_Coverage_Ratio'].mean():.2f} ± {n1_data['Duration_Coverage_Ratio'].std():.2f}\n")
    
    print(f"Summary statistics saved: {summary_file}")

def main():
    """메인 함수"""
    if len(sys.argv) != 2:
        print("Usage: python generate_heatmap.py <simulation_results_file_path>")
        print("Example: python generate_heatmap.py results/methods_comparison_20250601_134320_c067e8fa/simulation_results_20250601_134320.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # 입력 파일 확인
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    # 출력 디렉토리 설정 (입력 파일과 같은 디렉토리)
    output_dir = os.path.dirname(input_file)
    
    print(f"Parsing simulation results from: {input_file}")
    print(f"Output directory: {output_dir}")
    
    try:
        # 데이터 파싱
        df = parse_simulation_results(input_file)
        print(f"Successfully parsed {len(df)} simulation results")
        
        if len(df) == 0:
            print("Error: No data found in the file")
            sys.exit(1)
        
        # 히트맵 생성
        metrics = ['Infection_Coverage_Ratio', 'Infection_Duration', 'Duration_Coverage_Ratio']
        
        print("\n=== Creating combined heatmaps ===")
        create_heatmaps_by_n1(df, output_dir, metrics)
        
        print("\n=== Creating individual heatmaps ===")
        create_individual_heatmaps(df, output_dir, metrics)
        
        print("\n=== Generating summary statistics ===")
        generate_summary_stats(df, output_dir)
        
        print(f"\n=== All heatmaps and statistics generated successfully! ===")
        print(f"Check the output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 