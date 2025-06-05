import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import re
from datetime import datetime
from utils.utils import (load_saved_adjacency_matrix, generate_init_infected)
from models.epidemic_simulator_test import EpidemicSimulator

def create_heatmaps_from_results(results_file, output_dir):
    """
    시뮬레이션 결과로부터 히트맵 생성 (조합된 히트맵만)
    
    Args:
        results_file: 결과 파일 경로
        output_dir: 출력 디렉토리
    """
    print("\n=== Creating Heatmaps from Simulation Results ===")
    
    try:
        # 결과 파일 파싱
        df = parse_iteration_simulation_results(results_file)
        print(f"Successfully parsed {len(df)} method combinations with iteration results")
        
        if len(df) == 0:
            print("Error: No data found in the file")
            return
        
        # 히트맵 생성 (평균값, 조합된 것만)
        mean_metrics = ['Infection_Coverage_Ratio_Mean', 'Infection_Duration_Mean', 'Duration_Coverage_Ratio_Mean']
        
        print("\n=== Creating combined heatmaps (Mean Values) ===")
        create_heatmaps_by_n1_with_mean(df, output_dir, mean_metrics)
        
        print("\n=== Generating summary statistics ===")
        generate_summary_stats_with_iterations(df, output_dir)
        
        print(f"\n=== All heatmaps and statistics generated successfully! ===")
        print("Generated files include:")
        print("- Combined mean value heatmaps for each metric")
        print("- Summary statistics with iteration information")
        
    except Exception as e:
        print(f"Error during heatmap generation: {e}")

def parse_iteration_simulation_results(file_path):
    """
    여러 iteration 시뮬레이션 결과 파일을 파싱하여 DataFrame으로 변환 (평균값과 표준편차 포함)
    
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
        if "Method(N1,N2,N3)" in line and "Mean" in line:
            data_start = True
            continue
        
        # 구분선 건너뛰기
        if "---" in line or not line:
            continue
            
        # 데이터 라인 파싱
        if data_start and line.startswith('('):
            # (2, 4, 8)	0.5778	0.0234	28.50	2.14	52.30	3.45	2.9400	0.1234 형태 파싱
            parts = line.split('\t')
            if len(parts) >= 9:
                # Method 파싱: (2, 4, 8) -> [2, 4, 8]
                method_str = parts[0]
                method_match = re.findall(r'\d+', method_str)
                if len(method_match) == 3:
                    n1, n2, n3 = map(int, method_match)
                    
                    # 나머지 값들 파싱 (평균값만 사용, 표준편차는 별도 저장)
                    infection_coverage_ratio_mean = float(parts[1])
                    infection_coverage_ratio_std = float(parts[2])
                    infection_duration_mean = float(parts[3])
                    infection_duration_std = float(parts[4])
                    ever_infected_count_mean = float(parts[5])
                    ever_infected_count_std = float(parts[6])
                    r0_mean = float(parts[7])
                    r0_std = float(parts[8])
                    
                    # Duration/Coverage 비율 계산 (평균값 사용)
                    duration_coverage_ratio_mean = infection_duration_mean / infection_coverage_ratio_mean if infection_coverage_ratio_mean > 0 else 0
                    
                    data.append({
                        'N1': n1,
                        'N2': n2,
                        'N3': n3,
                        'Infection_Coverage_Ratio_Mean': infection_coverage_ratio_mean,
                        'Infection_Coverage_Ratio_Std': infection_coverage_ratio_std,
                        'Infection_Duration_Mean': infection_duration_mean,
                        'Infection_Duration_Std': infection_duration_std,
                        'Duration_Coverage_Ratio_Mean': duration_coverage_ratio_mean,
                        'Ever_Infected_Count_Mean': ever_infected_count_mean,
                        'Ever_Infected_Count_Std': ever_infected_count_std,
                        'R0_Mean': r0_mean,
                        'R0_Std': r0_std
                    })
    
    return pd.DataFrame(data)

def create_heatmaps_by_n1_with_mean(df, output_dir, metrics=['Infection_Coverage_Ratio_Mean', 'Infection_Duration_Mean', 'Duration_Coverage_Ratio_Mean']):
    """
    N1값별로 (N2, N3) 히트맵 생성 (평균값 사용, 조합된 것만, 통일된 색상 척도)
    
    Args:
        df: 파싱된 데이터프레임
        output_dir: 출력 디렉토리
        metrics: 히트맵을 생성할 지표들 (평균값)
    """
    # N1의 고유값들 찾기
    n1_values = sorted(df['N1'].unique())
    n2_values = sorted(df['N2'].unique())
    n3_values = sorted(df['N3'].unique())
    
    print(f"Found N1 values: {n1_values}")
    print(f"Found N2 values: {n2_values}")
    print(f"Found N3 values: {n3_values}")
    
    for metric in metrics:
        print(f"\nCreating combined heatmap for {metric}...")
        
        # 전체 데이터의 최대/최소값 계산 (색상 척도 통일을 위해)
        global_min = df[metric].min()
        global_max = df[metric].max()
        print(f"  Global range for {metric}: {global_min:.4f} - {global_max:.4f}")
        
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
            
            # 히트맵 생성 (통일된 색상 척도 사용)
            fmt_str = '.3f' if 'Ratio' in metric else '.1f'
            if metric == 'Duration_Coverage_Ratio_Mean':
                fmt_str = '.1f'
            
            sns.heatmap(
                pivot_table,
                annot=True,
                fmt=fmt_str,
                cmap='YlOrRd',
                ax=axes[i],
                vmin=global_min,  # 통일된 최소값
                vmax=global_max,  # 통일된 최대값
                cbar_kws={'label': metric.replace('_Mean', '')}
            )
            
            axes[i].set_title(f'N1 = {n1_val} (Mean)')
            axes[i].set_xlabel('N2 (Contact Type 2)')
            axes[i].set_ylabel('N3 (Contact Type 3)')
        
        plt.tight_layout()
        
        # 파일 저장
        metric_name = metric.replace('_Mean', '')
        output_file = os.path.join(output_dir, f'heatmap_{metric_name}_mean_by_N1.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def generate_summary_stats_with_iterations(df, output_dir):
    """
    여러 iteration 결과의 요약 통계 생성 및 저장
    
    Args:
        df: 파싱된 데이터프레임
        output_dir: 출력 디렉토리
    """
    summary_file = os.path.join(output_dir, 'summary_statistics_iterations.txt')
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== Simulation Results Summary Statistics (Multiple Iterations) ===\n\n")
        
        f.write(f"Total number of method combinations: {len(df)}\n")
        f.write(f"N1 values: {sorted(df['N1'].unique())}\n")
        f.write(f"N2 values: {sorted(df['N2'].unique())}\n")
        f.write(f"N3 values: {sorted(df['N3'].unique())}\n\n")
        
        f.write("=== Infection Coverage Ratio Statistics (Mean Values) ===\n")
        f.write(f"Overall Mean: {df['Infection_Coverage_Ratio_Mean'].mean():.4f}\n")
        f.write(f"Overall Std:  {df['Infection_Coverage_Ratio_Mean'].std():.4f}\n")
        f.write(f"Min:  {df['Infection_Coverage_Ratio_Mean'].min():.4f}\n")
        f.write(f"Max:  {df['Infection_Coverage_Ratio_Mean'].max():.4f}\n")
        f.write(f"Average Standard Deviation across iterations: {df['Infection_Coverage_Ratio_Std'].mean():.4f}\n\n")
        
        f.write("=== Infection Duration Statistics (Mean Values) ===\n")
        f.write(f"Overall Mean: {df['Infection_Duration_Mean'].mean():.2f}\n")
        f.write(f"Overall Std:  {df['Infection_Duration_Mean'].std():.2f}\n")
        f.write(f"Min:  {df['Infection_Duration_Mean'].min():.2f}\n")
        f.write(f"Max:  {df['Infection_Duration_Mean'].max():.2f}\n")
        f.write(f"Average Standard Deviation across iterations: {df['Infection_Duration_Std'].mean():.2f}\n\n")
        
        f.write("=== Duration/Coverage Ratio Statistics (Mean Values) ===\n")
        f.write(f"Overall Mean: {df['Duration_Coverage_Ratio_Mean'].mean():.2f}\n")
        f.write(f"Overall Std:  {df['Duration_Coverage_Ratio_Mean'].std():.2f}\n")
        f.write(f"Min:  {df['Duration_Coverage_Ratio_Mean'].min():.2f}\n")
        f.write(f"Max:  {df['Duration_Coverage_Ratio_Mean'].max():.2f}\n\n")
        
        # N1별 통계
        f.write("=== Statistics by N1 (Mean Values) ===\n")
        for n1_val in sorted(df['N1'].unique()):
            n1_data = df[df['N1'] == n1_val]
            f.write(f"\nN1 = {n1_val}:\n")
            f.write(f"  Infection Coverage Ratio (Mean): {n1_data['Infection_Coverage_Ratio_Mean'].mean():.4f} ± {n1_data['Infection_Coverage_Ratio_Mean'].std():.4f}\n")
            f.write(f"  Infection Duration (Mean): {n1_data['Infection_Duration_Mean'].mean():.2f} ± {n1_data['Infection_Duration_Mean'].std():.2f}\n")
            f.write(f"  Duration/Coverage Ratio (Mean): {n1_data['Duration_Coverage_Ratio_Mean'].mean():.2f} ± {n1_data['Duration_Coverage_Ratio_Mean'].std():.2f}\n")
            f.write(f"  Average Std across iterations - Coverage: {n1_data['Infection_Coverage_Ratio_Std'].mean():.4f}\n")
            f.write(f"  Average Std across iterations - Duration: {n1_data['Infection_Duration_Std'].mean():.2f}\n")
    
    print(f"Summary statistics saved: {summary_file}")

def run_repeated_simulations_with_iterations():
    """방역 대책별 반복 시뮬레이션 실행 (고정된 초기 감염자 조합으로 공정한 비교)"""
    print("=== Repeated Epidemic Simulation with Fixed Initial Infected Sets ===")
    
    try:
        # 1. config/config_test_statistics.json 읽기
        with open("config/config_test_statistics.json", 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config_params = config_data.get("defaults", {})
        
        # 2. iteration 횟수 설정 (config에서 읽어오거나 기본값 10)
        iterations = config_params.get('iterations', 10)
        
        # 3. 매트릭스 파일 경로 만들기
        matrix_name = config_params["save_matrix"]
        if not matrix_name.endswith('.json'):
            matrix_name += '.json'
        matrix_path = os.path.join("saved_matrix", matrix_name)
        
        # 4. 매트릭스 로드
        adj_matrix = load_saved_adjacency_matrix(matrix_path)
        
        # 5. 매트릭스 크기로 N 설정
        config_params['N'] = adj_matrix.shape[0]
        
        # 6. limit_starting_step 설정 (config에 없으면 기본값 10)
        config_params['limit_starting_step'] = config_params.get('limit_starting_step', 10)
        
        # 7. 방역 대책들 정의 (config에서 읽어오기)
        if 'methods' in config_params and config_params['methods']:
            prevention_methods = [tuple(method) for method in config_params['methods']]
        else:
            # 기본 방역 대책들 (config에 methods가 없는 경우)
            prevention_methods = [
                (0, 2, 3),  # n1 차단, n2,n3 유지
                (1, 1, 2),  # n1,n2 감소, n3 감소
                (1, 2, 2),  # n1 감소, n2 유지, n3 감소
                (config_params['N1'], config_params['N2'], config_params['N3'])  # 기준점 (방역 없음)
            ]
        
        # 8. 통합 결과 폴더 생성
        temp_simulator = EpidemicSimulator(config_params)
        result_path = temp_simulator.create_result_folder("methods_comparison_iterations")
        
        # 9. 결과 저장을 위한 파일 준비
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(result_path, f"simulation_results_iterations_{timestamp}.txt")
        
        # 10. 초기 감염자 조합들을 미리 생성 (모든 method에 공통으로 사용)
        print("=== Generating fixed initial infected sets ===")
        init_infected_list = []
        for i in range(iterations):
            np.random.seed(config_params['seed'] + i)
            init_infected = generate_init_infected(config_params)
            init_infected_list.append(init_infected)
            print(f"Initial infected set {i+1}: {init_infected}")
        
        # 11. 각 method별로 결과를 저장할 딕셔너리 (리스트로 초기화)
        method_results = {}
        for method in prevention_methods:
            method_results[method] = []
        
        print(f"\nResults will be saved to: {result_path}")
        print(f"Network size: {config_params['N']} nodes")
        print(f"Initial infected sets: {iterations}")
        print(f"Total methods: {len(prevention_methods)}")
        print(f"Total simulations: {iterations * len(prevention_methods)}")
        print(f"Limit starting step: {config_params['limit_starting_step']}")
        print()
        
        # 12. 각 초기 감염자 조합에 대해 모든 method 테스트
        for i, init_infected in enumerate(init_infected_list):
            print(f"\n=== Initial Infected Set {i+1}/{iterations}: {init_infected} ===")
            
            for method_idx, method in enumerate(prevention_methods):
                print(f"  Method {method_idx+1}/{len(prevention_methods)}: {method}")
                
                # 시뮬레이션 실행
                simulator = EpidemicSimulator(
                    config_params=config_params, 
                    adj_matrix=adj_matrix, 
                    init_infected=init_infected,
                    method=method
                )
                
                # result_path를 미리 설정
                simulator.result_path = result_path
                
                # 시뮬레이션 실행 (결과 저장 비활성화)
                result = simulator.run_simulation(save_results=False, create_plots=False)
                
                # 사용자 정의 지표 계산
                custom_metrics = simulator.calculate_custom_metrics()
                
                # 이번 iteration 결과를 method별로 저장
                iteration_result = {
                    'r0': result['r0'],
                    'infection_coverage_ratio': custom_metrics['infection_coverage_ratio'],
                    'infection_duration': custom_metrics['infection_duration'],
                    'ever_infected_count': custom_metrics['ever_infected_count']
                }
                method_results[method].append(iteration_result)
        
        # 13. 각 method별로 평균과 표준편차 계산
        print(f"\n=== Calculating statistics for each method ===")
        final_method_results = {}
        
        for method in prevention_methods:
            results_list = method_results[method]
            
            avg_result = {
                'method': method,
                'r0_mean': np.mean([r['r0'] for r in results_list]),
                'r0_std': np.std([r['r0'] for r in results_list]),
                'infection_coverage_ratio_mean': np.mean([r['infection_coverage_ratio'] for r in results_list]),
                'infection_coverage_ratio_std': np.std([r['infection_coverage_ratio'] for r in results_list]),
                'infection_duration_mean': np.mean([r['infection_duration'] for r in results_list]),
                'infection_duration_std': np.std([r['infection_duration'] for r in results_list]),
                'ever_infected_count_mean': np.mean([r['ever_infected_count'] for r in results_list]),
                'ever_infected_count_std': np.std([r['ever_infected_count'] for r in results_list]),
                'iterations': iterations
            }
            
            final_method_results[method] = avg_result
            
            # 진행 상황 출력
            print(f"  Method {method}:")
            print(f"    R0: {avg_result['r0_mean']:.3f} ± {avg_result['r0_std']:.3f}")
            print(f"    Coverage: {avg_result['infection_coverage_ratio_mean']:.3f} ± {avg_result['infection_coverage_ratio_std']:.3f}")
            print(f"    Duration: {avg_result['infection_duration_mean']:.1f} ± {avg_result['infection_duration_std']:.1f}")
        
        # 14. 결과 파일에 헤더 작성
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=== Epidemic Simulation Results with Fixed Initial Infected Sets ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Matrix file: {matrix_path}\n")
            f.write(f"Network size: {config_params['N']} nodes\n")
            f.write(f"Initial infected sets: {iterations}\n")
            f.write(f"Limit starting step: {config_params['limit_starting_step']}\n")
            f.write(f"Original contact rates (N1, N2, N3): ({config_params['N1']}, {config_params['N2']}, {config_params['N3']})\n")
            f.write("\nInitial infected sets used:\n")
            for i, init_set in enumerate(init_infected_list):
                f.write(f"  Set {i+1}: {init_set}\n")
            f.write("\n")
            f.write("Method(N1,N2,N3)\tInfection_Coverage_Ratio_Mean\tInfection_Coverage_Ratio_Std\t")
            f.write("Infection_Duration_Mean\tInfection_Duration_Std\tEver_Infected_Count_Mean\t")
            f.write("Ever_Infected_Count_Std\tR0_Mean\tR0_Std\n")
            f.write("-" * 150 + "\n")
            
            # 결과 데이터 작성
            for method, result in final_method_results.items():
                f.write(f"{method}\t{result['infection_coverage_ratio_mean']:.4f}\t{result['infection_coverage_ratio_std']:.4f}\t")
                f.write(f"{result['infection_duration_mean']:.2f}\t{result['infection_duration_std']:.2f}\t")
                f.write(f"{result['ever_infected_count_mean']:.2f}\t{result['ever_infected_count_std']:.2f}\t")
                f.write(f"{result['r0_mean']:.4f}\t{result['r0_std']:.4f}\n")
        
        print(f"\n=== All simulations completed ===")
        print(f"Results saved to: {result_path}")
        print(f"Total methods tested: {len(prevention_methods)}")
        print(f"Initial infected sets: {iterations}")
        print(f"Total simulations performed: {iterations * len(prevention_methods)}")
        
        # 요약 정보 출력
        print("\nSummary (Mean ± Std):")
        for method, result in final_method_results.items():
            print(f"  {method}: Coverage={result['infection_coverage_ratio_mean']:.3f}±{result['infection_coverage_ratio_std']:.3f}, "
                  f"Duration={result['infection_duration_mean']:.1f}±{result['infection_duration_std']:.1f}")
    
        # 15. 히트맵 자동 생성 (조합된 것만)
        print(f"\n=== Starting automatic heatmap generation ===")
        create_heatmaps_from_results(results_file, result_path)
        
        print(f"\n=== All processes completed! ===")
        print(f"Check the results directory: {result_path}")
        print("Generated files include:")
        print("- Simulation results with iteration statistics")
        print("- Combined mean value heatmaps for each metric")
        print("- Summary statistics")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("\n사용 가능한 매트릭스 파일:")
        if os.path.exists("saved_matrix"):
            files = [f for f in os.listdir("saved_matrix") if f.endswith('.json')]
            for f in files:
                print(f"  - {f.replace('.json', '')}")
        print("\nconfig/config_test_statistics.json에서 \"save_matrix\": \"파일명\"으로 설정하세요.")

def main():
    print("=== Main ===")

if __name__ == "__main__":
    import sys
    run_repeated_simulations_with_iterations() 