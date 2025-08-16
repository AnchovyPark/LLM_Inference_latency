"""
GPU별 현실적인 모델 조합 시나리오 생성기

각 GPU의 메모리 제약을 고려하여 실제로 실행 가능한 
모델 조합들을 생성합니다.
"""

import csv
import random
import os
from typing import List, Dict, Tuple

# GPU별 현실적인 모델 조합 (메모리 제약 고려)
GPU_MODEL_COMBINATIONS = {
    'T4': ['LLaMA_3.2_1B'],  # 16GB - 작은 모델만
    'L4': ['LLaMA_3.2_1B', 'LLaMA_3_8B', 'LLaMA_3.1_8B'],  # 24GB - 중간 모델까지
    'A10G': ['LLaMA_3.2_1B', 'LLaMA_3_8B', 'LLaMA_3.1_8B'],  # 24GB - 중간 모델까지  
    'L40S': ['LLaMA_3_8B', 'LLaMA_3.1_8B', 'LLaMA_3_70B'],  # 48GB - 큰 모델까지
    'A100': ['LLaMA_3_8B', 'LLaMA_3.1_8B', 'LLaMA_3_70B'],  # 40GB - 큰 모델까지
    'A100-SXM4-80GB': ['LLaMA_3.1_8B', 'LLaMA_3_70B']  # 80GB - 가장 큰 모델 전용
}

# 입력/출력 길이 시나리오
SEQUENCE_SCENARIOS = [
    # 짧은 대화
    {'input_range': (100, 512), 'output_range': (50, 200), 'scenario': 'short_chat'},
    # 중간 문서 처리  
    {'input_range': (512, 2048), 'output_range': (200, 500), 'scenario': 'medium_doc'},
    # 긴 문서 분석
    {'input_range': (2048, 4096), 'output_range': (500, 1000), 'scenario': 'long_analysis'},
    # 코드 생성
    {'input_range': (200, 1000), 'output_range': (100, 800), 'scenario': 'code_gen'}
]


def generate_realistic_scenarios(num_scenarios: int = 100) -> List[Dict]:
    """현실적인 GPU-모델-시퀀스 조합 시나리오 생성"""
    
    scenarios = []
    scenario_id = 1
    
    for _ in range(num_scenarios):
        # GPU 선택
        gpu = random.choice(list(GPU_MODEL_COMBINATIONS.keys()))
        
        # 해당 GPU에서 실행 가능한 모델 선택
        available_models = GPU_MODEL_COMBINATIONS[gpu]
        model = random.choice(available_models)
        
        # 시퀀스 길이 시나리오 선택
        seq_scenario = random.choice(SEQUENCE_SCENARIOS)
        
        # 입력/출력 길이 랜덤 생성
        input_length = random.randint(*seq_scenario['input_range'])
        output_length = random.randint(*seq_scenario['output_range'])
        
        # 배치 사이즈 (대부분 1, 가끔 2-4)
        batch_size = random.choices([1, 2, 4], weights=[0.8, 0.15, 0.05])[0]
        
        scenario = {
            'scenario_id': scenario_id,
            'gpu': gpu,
            'model': model,
            'input_length': input_length,
            'output_length': output_length,
            'batch_size': batch_size,
            'use_case': seq_scenario['scenario'],
            'description': f"{gpu}에서 {model} 모델로 {seq_scenario['scenario']} 시나리오"
        }
        
        scenarios.append(scenario)
        scenario_id += 1
    
    return scenarios


def generate_balanced_scenarios() -> List[Dict]:
    """각 GPU별로 균등하게 배분된 시나리오 생성"""
    
    scenarios = []
    scenario_id = 1
    
    # 각 GPU별로 동일한 수의 시나리오 생성
    scenarios_per_gpu = 20
    
    for gpu, available_models in GPU_MODEL_COMBINATIONS.items():
        for _ in range(scenarios_per_gpu):
            # 모델 선택
            model = random.choice(available_models)
            
            # 시퀀스 시나리오 선택
            seq_scenario = random.choice(SEQUENCE_SCENARIOS)
            
            # 입력/출력 길이
            input_length = random.randint(*seq_scenario['input_range'])
            output_length = random.randint(*seq_scenario['output_range'])
            
            # 배치 사이즈
            batch_size = random.choices([1, 2, 4], weights=[0.8, 0.15, 0.05])[0]
            
            scenario = {
                'scenario_id': scenario_id,
                'gpu': gpu,
                'model': model, 
                'input_length': input_length,
                'output_length': output_length,
                'batch_size': batch_size,
                'use_case': seq_scenario['scenario'],
                'description': f"{gpu}에서 {model} 모델로 {seq_scenario['scenario']} 시나리오"
            }
            
            scenarios.append(scenario)
            scenario_id += 1
    
    return scenarios


def save_scenarios_to_csv(scenarios: List[Dict], filename: str):
    """시나리오를 CSV 파일로 저장"""
    
    fieldnames = ['scenario_id', 'gpu', 'model', 'input_length', 'output_length', 
                  'batch_size', 'use_case', 'description']
    
    os.makedirs('/Users/anchovy-mac/Desktop/calculating/experiment', exist_ok=True)
    filepath = f'/Users/anchovy-mac/Desktop/calculating/experiment/{filename}'
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenarios)
    
    print(f"💾 {len(scenarios)}개 시나리오가 {filepath}에 저장되었습니다")


def print_scenario_summary(scenarios: List[Dict]):
    """시나리오 요약 출력"""
    
    print("📊 시나리오 요약:")
    print("-" * 60)
    
    # GPU별 통계
    gpu_counts = {}
    model_counts = {}
    use_case_counts = {}
    
    for scenario in scenarios:
        gpu = scenario['gpu']
        model = scenario['model']
        use_case = scenario['use_case']
        
        gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
        model_counts[model] = model_counts.get(model, 0) + 1  
        use_case_counts[use_case] = use_case_counts.get(use_case, 0) + 1
    
    print("🖥️  GPU별 시나리오 수:")
    for gpu, count in gpu_counts.items():
        print(f"  • {gpu}: {count}개")
    
    print(f"\n🤖 모델별 시나리오 수:")
    for model, count in model_counts.items():
        print(f"  • {model}: {count}개")
    
    print(f"\n📝 사용 사례별 시나리오 수:")
    for use_case, count in use_case_counts.items():
        print(f"  • {use_case}: {count}개")
    
    print(f"\n📈 총 시나리오 수: {len(scenarios)}개")


if __name__ == "__main__":
    print("🚀 LLM 지연시간 예측 시나리오 생성기")
    print("=" * 50)
    
    # 1. 랜덤 시나리오 생성 (100개)
    print("\n1️⃣  랜덤 시나리오 생성 중...")
    random_scenarios = generate_realistic_scenarios(100)
    save_scenarios_to_csv(random_scenarios, 'random_scenarios.csv')
    
    # 2. 균등 배분 시나리오 생성 
    print("\n2️⃣  균등 배분 시나리오 생성 중...")
    balanced_scenarios = generate_balanced_scenarios()
    save_scenarios_to_csv(balanced_scenarios, 'balanced_scenarios.csv')
    
    # 3. 요약 출력
    print("\n📊 균등 배분 시나리오 요약:")
    print_scenario_summary(balanced_scenarios)
    
    print(f"\n✅ 시나리오 생성 완료!")
    print(f"📁 파일 위치: /Users/anchovy-mac/Desktop/calculating/experiment/")