import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import os
import uuid
from datetime import datetime
from tqdm import tqdm
from itertools import product
from utils import Config
from epidemic_simulator import EpidemicSimulator

def create_experiment_folder():
    """Create unique experiment folder with timestamp and UUID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    folder_name = f"experiment_{timestamp}_{unique_id}"
    
    if not os.path.exists("results"):
        os.makedirs("results")
    
    experiment_path = os.path.join("results", folder_name)
    os.makedirs(experiment_path)
    
    return experiment_path

def run_parameter_grid_experiment(
    base_config: Config,
    param_grid: dict,
    init_infected_ratio: float = 0.01,
    repetitions: int = 5
):
    """
    다중 파라미터 조합 실험 실행
    
    Args:
        base_config: 기본 설정 객체
        param_grid: 실험할 파라미터 조합 dict (예: {'tau': [0.1, 0.3], 'alpha': [0.05, 0.1]})
        init_infected_ratio: 초기 감염 비율
        repetitions: 각 조합당 반복 횟수
    
    Returns:
        pandas.DataFrame: 실험 결과 요약
    """
    # 모든 파라미터 조합 생성
    param_keys = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))
    param_dicts = [dict(zip(param_keys, values)) for values in param_combinations]
    
    print(f"Running grid experiment with {len(param_dicts)} parameter combinations")
    print(f"Parameters: {param_keys}")
    print(f"Repetitions per combination: {repetitions}")
    
    results_list = []
    
    for i, param_set in enumerate(tqdm(param_dicts, desc="Parameter combinations")):
        # 각 조합에 대해 반복 실험
        rep_metrics = []
        
        for rep in range(repetitions):
            # 설정 생성
            config_dict = {
                'N': base_config.N,
                'N1': base_config.N1,
                'N2': base_config.N2,
                'N3': base_config.N3,
                'tau': base_config.tau,
                'alpha': base_config.alpha,
                'xi': base_config.xi,
                'T': base_config.T,
                'seed': base_config.seed + rep if base_config.seed else None,
                # 실험에서는 애니메이션 비활성화
                'enable_animation': False,
                'save_mp4': False,
                'save_gif': False,
                'show_animation': False
            }
            
            # 특정 파라미터 업데이트
            config_dict.update(param_set)
            config = Config(**config_dict)
            
            # 초기 감염자 설정
            n_infected = max(1, int(config.N * init_infected_ratio))
            init_infected = list(np.random.choice(config.N, n_infected, replace=False))
            
            # 시뮬레이션 실행
            simulator = EpidemicSimulator(config, init_infected=init_infected)
            sim_results = simulator.run_simulation(save_results=False, create_plots=False)
            
            # 핵심 지표 추출
            stats = sim_results['statistics']
            metrics = {
                'r0': sim_results['r0'],
                'peak_infected_ratio': stats['peak_infected_ratio'],
                'final_recovered_ratio': stats['final_recovered'] / config.N,
                'epidemic_duration': stats['epidemic_duration'],
                'attack_rate': stats['attack_rate']
            }
            rep_metrics.append(metrics)
        
        # 반복 실험 결과 평균 및 표준편차 계산
        row = {**param_set}  # 파라미터 조합 추가
        
        for metric_name in rep_metrics[0].keys():
            values = [m[metric_name] for m in rep_metrics]
            row[f"{metric_name}_mean"] = np.mean(values)
            row[f"{metric_name}_std"] = np.std(values)
        
        results_list.append(row)
    
    return pd.DataFrame(results_list)

def create_experiment_visualizations(df, experiment_path):
    """핵심 시각화 생성"""
    
    # 1. R0 vs Peak Infected 산점도
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['r0_mean'], df['peak_infected_ratio_mean'], 
                alpha=0.7, s=80, c=df['attack_rate_mean'], cmap='viridis')
    plt.colorbar(label='Attack Rate')
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='R0=1 threshold')
    plt.xlabel('R0 (Basic Reproduction Number)')
    plt.ylabel('Peak Infected Ratio')
    plt.title('R0 vs Peak Infection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 파라미터별 R0 분포 (박스플롯)
    plt.subplot(1, 2, 2)
    
    # 주요 파라미터 중 하나 선택 (tau가 있으면 tau 사용)
    if 'tau' in df.columns:
        main_param = 'tau'
    elif 'alpha' in df.columns:
        main_param = 'alpha'
    else:
        main_param = df.columns[0]  # 첫 번째 파라미터
    
    unique_vals = sorted(df[main_param].unique())
    r0_groups = [df[df[main_param] == val]['r0_mean'].values for val in unique_vals]
    
    plt.boxplot(r0_groups, labels=unique_vals)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    plt.xlabel(f'{main_param.upper()} Value')
    plt.ylabel('R0 Distribution')
    plt.title(f'R0 Distribution by {main_param.upper()}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, 'experiment_summary.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 상관관계 히트맵 (파라미터가 3개 이상일 때)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 3:
        plt.figure(figsize=(10, 8))
        
        # 주요 지표들만 선택
        key_metrics = [col for col in numeric_cols if any(metric in col for metric in 
                      ['r0_mean', 'peak_infected_ratio_mean', 'attack_rate_mean', 'tau', 'alpha', 'xi'])]
        
        if len(key_metrics) >= 3:
            corr_matrix = df[key_metrics].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title('Parameter and Outcome Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(experiment_path, 'correlation_heatmap.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def analyze_results(df, experiment_path):
    """결과 분석 및 요약 생성"""
    
    analysis_text = f"""SIRS Model Parameter Grid Experiment Analysis

=== Experiment Overview ===
- Total parameter combinations tested: {len(df)}
- Parameter ranges tested:
"""
    
    # 파라미터 범위 정보
    param_cols = [col for col in df.columns if col not in 
                 [col for col in df.columns if '_mean' in col or '_std' in col]]
    
    for param in param_cols:
        if df[param].dtype in ['float64', 'int64']:
            analysis_text += f"  - {param}: {df[param].min():.3f} to {df[param].max():.3f}\n"
    
    analysis_text += f"""
=== Key Findings ===
- Epidemic threshold (R0 > 1): {(df['r0_mean'] > 1).sum()}/{len(df)} combinations
- Highest R0 achieved: {df['r0_mean'].max():.3f}
- Highest peak infection: {df['peak_infected_ratio_mean'].max():.1%}
- Highest attack rate: {df['attack_rate_mean'].max():.1%}

=== Parameter Impact Rankings ===
"""
    
    # 상관관계 분석
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outcome_metrics = ['r0_mean', 'peak_infected_ratio_mean', 'attack_rate_mean']
    param_cols_numeric = [col for col in numeric_cols if col not in 
                         [col for col in numeric_cols if '_mean' in col or '_std' in col]]
    
    for outcome in outcome_metrics:
        if outcome in df.columns:
            analysis_text += f"\nCorrelations with {outcome}:\n"
            correlations = df[param_cols_numeric + [outcome]].corr()[outcome].drop(outcome)
            sorted_corr = correlations.abs().sort_values(ascending=False)
            
            for param, corr_val in sorted_corr.items():
                analysis_text += f"  - {param}: {correlations[param]:+.3f}\n"
    
    analysis_text += f"""
=== Files Generated ===
- experiment_results.csv: Complete experimental data
- experiment_summary.png: Key visualizations
- correlation_heatmap.png: Parameter correlation analysis (if applicable)
- experiment_analysis.txt: This analysis file
"""
    
    with open(os.path.join(experiment_path, 'experiment_analysis.txt'), 'w') as f:
        f.write(analysis_text)

def run_comprehensive_experiments():
    """포괄적인 파라미터 그리드 실험 실행"""
    
    # 실험 폴더 생성
    experiment_path = create_experiment_folder()
    print(f"Experiment results will be saved to: {experiment_path}")
    
    # 기본 설정
    base_cfg = Config(
        N=200,       
        N1=3,        
        N2=2,        
        N3=1,        
        tau=0.3,     # 기본값 (실험에서 변경됨)
        alpha=0.1,   # 기본값 (실험에서 변경됨)
        xi=0.05,     # 기본값 (실험에서 변경됨)
        T=100,       
        seed=42,     
        enable_animation=False  # 실험에서는 애니메이션 비활성화
    )
    
    # 파라미터 그리드 정의
    # param_grid = {
    #     'tau': [0.1, 0.2, 0.3, 0.4, 0.5],           # 감염 확률
    #     'alpha': [0.05, 0.1, 0.15, 0.2],            # 회복 확률
    #     'xi': [0.01, 0.03, 0.05, 0.07, 0.1]         # 면역 상실 확률
    # }
    param_grid = {
        'tau': [0.3, 0.4],           # 감염 확률
        'alpha': [0.1],            # 회복 확률
        'xi': [0.05, 0.07]         # 면역 상실 확률
    }
    
    print(f"Parameter grid:")
    for param, values in param_grid.items():
        print(f"  - {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"Total combinations: {total_combinations}")
    
    # 실험 실행
    results_df = run_parameter_grid_experiment(
        base_cfg, param_grid, 
        repetitions=3,  # 계산 시간 단축을 위해 3회 반복
        init_infected_ratio=0.01
    )
    
    # 결과 저장
    results_df.to_csv(os.path.join(experiment_path, 'experiment_results.csv'), index=False)
    print(f"Results saved to: {os.path.join(experiment_path, 'experiment_results.csv')}")
    
    # 시각화 생성
    create_experiment_visualizations(results_df, experiment_path)
    print("Visualizations created")
    
    # 분석 결과 생성
    analyze_results(results_df, experiment_path)
    print("Analysis completed")
    
    print(f"\n{'='*60}")
    print("Grid experiment completed successfully!")
    print(f"Results saved to: {experiment_path}")
    print(f"Key files:")
    print(f"  - experiment_results.csv: Complete data")
    print(f"  - experiment_summary.png: Key plots")
    print(f"  - experiment_analysis.txt: Statistical analysis")
    print(f"{'='*60}")
    
    return results_df, experiment_path

if __name__ == "__main__":
    # 포괄적인 실험 실행
    results, path = run_comprehensive_experiments()
    
    # 간단한 결과 요약 출력
    print(f"\nQuick Summary:")
    print(f"- Tested {len(results)} parameter combinations")
    print(f"- R0 range: {results['r0_mean'].min():.2f} to {results['r0_mean'].max():.2f}")
    print(f"- Peak infection range: {results['peak_infected_ratio_mean'].min():.1%} to {results['peak_infected_ratio_mean'].max():.1%}")
    print(f"- Epidemic combinations (R0>1): {(results['r0_mean'] > 1).sum()}/{len(results)}") 