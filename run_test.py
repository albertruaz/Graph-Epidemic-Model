import json
import os
from datetime import datetime
from utils import (load_saved_adjacency_matrix, generate_init_infected)
from epidemic_simulator_test import EpidemicSimulator

def run_repeated_simulations():
    """방역 대책별 반복 시뮬레이션 실행"""
    print("=== Repeated Epidemic Simulation with Prevention Methods ===")
    
    try:
        # 1. config_test.json 읽기
        with open("config_test.json", 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        config_params = config_data.get("defaults", {})
        
        # 2. 매트릭스 파일 경로 만들기
        matrix_name = config_params["save_matrix"]
        if not matrix_name.endswith('.json'):
            matrix_name += '.json'
        matrix_path = os.path.join("saved_matrix", matrix_name)
        
        # 3. 매트릭스 로드
        adj_matrix = load_saved_adjacency_matrix(matrix_path)
        
        # 4. 매트릭스 크기로 N 설정
        config_params['N'] = adj_matrix.shape[0]
        
        # 5. limit_starting_step 설정 (config에 없으면 기본값 10)
        config_params['limit_starting_step'] = config_params.get('limit_starting_step', 10)
        
        # 6. 방역 대책들 정의 (config에서 읽어오기)
        if 'methods' in config_params and config_params['methods']:
            prevention_methods = [tuple(method) for method in config_params['methods']]
            # 기준점 (방역 없음)을 마지막에 추가
            prevention_methods.append((config_params['N1'], config_params['N2'], config_params['N3']))
        else:
            # 기본 방역 대책들 (config에 methods가 없는 경우)
            prevention_methods = [
                (0, 2, 3),  # n1 차단, n2,n3 유지
                (1, 1, 2),  # n1,n2 감소, n3 감소
                (1, 2, 2),  # n1 감소, n2 유지, n3 감소
                (config_params['N1'], config_params['N2'], config_params['N3'])  # 기준점 (방역 없음)
            ]
        
        # 7. 통합 결과 폴더 생성
        temp_simulator = EpidemicSimulator(config_params)
        result_path = temp_simulator.create_result_folder("methods_comparison")
        
        # 8. 결과 저장을 위한 파일 준비
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(result_path, f"simulation_results_{timestamp}.txt")
        
        # 헤더 작성
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("=== Epidemic Simulation Results with Prevention Methods ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Matrix file: {matrix_path}\n")
            f.write(f"Network size: {config_params['N']} nodes\n")
            f.write(f"Limit starting step: {config_params['limit_starting_step']}\n")
            f.write(f"Original contact rates (N1, N2, N3): ({config_params['N1']}, {config_params['N2']}, {config_params['N3']})\n")
            f.write("\n")
            f.write("Method(N1,N2,N3)\tInfection_Coverage_Ratio\tInfection_Duration\tEver_Infected_Count\tR0\n")
            f.write("-" * 100 + "\n")
        
        print(f"Results will be saved to: {result_path}")
        print(f"Network size: {config_params['N']} nodes")
        print(f"Limit starting step: {config_params['limit_starting_step']}")
        print()
        
        # 9. 각 방역 대책에 대해 시뮬레이션 실행
        for i, method in enumerate(prevention_methods):
            print(f"Running simulation {i+1}/{len(prevention_methods)}: Method {method}")
            
            # 초기 감염자 생성 (매번 동일하게)
            init_infected = generate_init_infected(config_params)
            
            # 시뮬레이션 실행 (같은 result_path 사용)
            simulator = EpidemicSimulator(
                config_params=config_params, 
                adj_matrix=adj_matrix, 
                init_infected=init_infected,
                method=method
            )
            
            # result_path를 미리 설정
            simulator.result_path = result_path
            
            # 시뮬레이션 실행 (결과 저장 비활성화하고 수동으로 필요한 것만 저장)
            result = simulator.run_simulation(save_results=False, create_plots=False)
            
            # SIR dynamics만 method별로 저장
            simulator._plot_sir_dynamics()
            
            # 사용자 정의 지표 계산
            custom_metrics = simulator.calculate_custom_metrics()
            
            # 결과 출력
            print(f"  - Method: {method}")
            print(f"  - R0: {result['r0']:.3f}")
            print(f"  - Infection Coverage Ratio: {custom_metrics['infection_coverage_ratio']:.3f}")
            print(f"  - Infection Duration: {custom_metrics['infection_duration']} steps")
            print(f"  - Ever Infected Count: {custom_metrics['ever_infected_count']}")
            print()
            
            # 결과를 파일에 추가
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(f"{method}\t{custom_metrics['infection_coverage_ratio']:.4f}\t")
                f.write(f"{custom_metrics['infection_duration']}\t{custom_metrics['ever_infected_count']}\t")
                f.write(f"{result['r0']:.4f}\n")
        
        print(f"=== All simulations completed ===")
        print(f"Results saved to: {result_path}")
        
        # 요약 정보 출력
        print("\nSummary:")
        with open(results_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('('):  # 방법별 결과 라인
                    print(f"  {line.strip()}")
    
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("\n사용 가능한 매트릭스 파일:")
        if os.path.exists("saved_matrix"):
            files = [f for f in os.listdir("saved_matrix") if f.endswith('.json')]
            for f in files:
                print(f"  - {f.replace('.json', '')}")
        print("\nconfig_test.json에서 \"save_matrix\": \"파일명\"으로 설정하세요.")

def main():
    print("=== Main ===")

if __name__ == "__main__":
    import sys
    run_repeated_simulations()
    