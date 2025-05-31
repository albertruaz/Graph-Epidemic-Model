import json
import os
from utils import (load_saved_adjacency_matrix, generate_init_infected)
from epidemic_simulator import EpidemicSimulator

def main():
    print("=== Epidemic Simulation ===")
    
    try:
        # 1. config.json 읽기
        with open("config.json", 'r', encoding='utf-8') as f:
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
        
        # 5. 초기 감염자 생성
        init_infected = generate_init_infected(config_params)
        
        # 6. 시뮬레이션 실행
        simulator = EpidemicSimulator(config_params, adj_matrix=adj_matrix, init_infected=init_infected)
        result = simulator.run_simulation(save_results=True, create_plots=True)
        
        # 결과 출력
        print(f"\n=== Simulation Results ===")
        print(f"Matrix file: {matrix_path}")
        print(f"Network size: {config_params['N']} nodes")
        print(f"Initial infected: {init_infected}")
        print(f"R0: {result['r0']:.2f}")
        print(f"Results saved to: {result['result_path']}")
        
    except Exception as e:
        print(f"Error during simulation: {e}")
        print("\n사용 가능한 매트릭스 파일:")
        if os.path.exists("saved_matrix"):
            files = [f for f in os.listdir("saved_matrix") if f.endswith('.json')]
            for f in files:
                print(f"  - {f.replace('.json', '')}")
        print("\nconfig.json에서 \"save_matrix\": \"파일명\"으로 설정하세요.")

if __name__ == "__main__":
    main()
    