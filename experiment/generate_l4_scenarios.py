"""
L4 GPU + LLaMA 3.1 8B 특화 시나리오 생성기

Input/Output 길이를 체계적으로 변경하며 성능 특성을 분석하기 위한
상세한 시나리오를 생성합니다.
"""

import csv
import os
from typing import List, Dict

def generate_l4_llama31_scenarios() -> List[Dict]:
    """L4 + LLaMA 3.1 8B 전용 체계적 시나리오 생성"""
    
    scenarios = []
    scenario_id = 1
    
    # 고정값들
    gpu = 'L4'
    model = 'LLaMA_3.1_8B'
    batch_size = 1
    
    # Input/Output 조합들 정의
    test_combinations = [
        # Input 512 시리즈
        {'input': 512, 'outputs': [32, 64, 128, 256, 512]},
        # Input 1024 시리즈  
        {'input': 1024, 'outputs': [32, 64, 128, 256, 512, 1024]},
        # Input 2048 시리즈 (추가 테스트)
        {'input': 2048, 'outputs': [32, 64, 128, 256, 512, 1024, 2048]},
        # Input 256 시리즈 (빠른 응답)
        {'input': 256, 'outputs': [32, 64, 128, 256]},
        # Input 4096 시리즈 (긴 컨텍스트)
        {'input': 4096, 'outputs': [32, 64, 128, 256, 512]}
    ]
    
    for combo in test_combinations:
        input_length = combo['input']
        
        for output_length in combo['outputs']:
            # 사용 사례 분류
            if input_length <= 512 and output_length <= 128:
                use_case = 'quick_qa'
                description = f"빠른 Q&A: {input_length}→{output_length} 토큰"
            elif input_length <= 1024 and output_length <= 256:
                use_case = 'chat_response'  
                description = f"채팅 응답: {input_length}→{output_length} 토큰"
            elif input_length <= 2048 and output_length <= 512:
                use_case = 'document_summary'
                description = f"문서 요약: {input_length}→{output_length} 토큰"
            elif output_length >= 512:
                use_case = 'content_generation'
                description = f"콘텐츠 생성: {input_length}→{output_length} 토큰"
            else:
                use_case = 'long_context'
                description = f"긴 컨텍스트: {input_length}→{output_length} 토큰"
            
            scenario = {
                'scenario_id': scenario_id,
                'gpu': gpu,
                'model': model,
                'input_length': input_length,
                'output_length': output_length,
                'batch_size': batch_size,
                'use_case': use_case,
                'description': description,
                'input_category': f'input_{input_length}',
                'output_category': f'output_{output_length}',
                'total_tokens': input_length + output_length
            }
            
            scenarios.append(scenario)
            scenario_id += 1
    
    return scenarios


def save_scenarios_to_csv(scenarios: List[Dict], filename: str):
    """시나리오를 CSV 파일로 저장"""
    
    fieldnames = [
        'scenario_id', 'gpu', 'model', 'input_length', 'output_length', 
        'batch_size', 'use_case', 'description', 'input_category', 
        'output_category', 'total_tokens'
    ]
    
    os.makedirs('/Users/anchovy-mac/Desktop/calculating/experiment', exist_ok=True)
    filepath = f'/Users/anchovy-mac/Desktop/calculating/experiment/{filename}'
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scenarios)
    
    print(f"💾 {len(scenarios)}개 L4 시나리오가 {filepath}에 저장되었습니다")


def analyze_scenarios(scenarios: List[Dict]):
    """생성된 시나리오 분석"""
    
    print("\n📊 L4 + LLaMA 3.1 8B 시나리오 분석:")
    print("-" * 60)
    
    # Input 길이별 통계
    input_stats = {}
    output_stats = {}
    use_case_stats = {}
    
    for scenario in scenarios:
        input_len = scenario['input_length']
        output_len = scenario['output_length']
        use_case = scenario['use_case']
        
        input_stats[input_len] = input_stats.get(input_len, 0) + 1
        output_stats[output_len] = output_stats.get(output_len, 0) + 1
        use_case_stats[use_case] = use_case_stats.get(use_case, 0) + 1
    
    print(f"📥 Input 길이별 시나리오 수:")
    for input_len in sorted(input_stats.keys()):
        print(f"  • {input_len} 토큰: {input_stats[input_len]}개")
    
    print(f"\n📤 Output 길이별 시나리오 수:")
    for output_len in sorted(output_stats.keys()):
        print(f"  • {output_len} 토큰: {output_stats[output_len]}개")
    
    print(f"\n🎯 사용 사례별 시나리오 수:")
    for use_case, count in use_case_stats.items():
        print(f"  • {use_case}: {count}개")
    
    print(f"\n📈 총 시나리오 수: {len(scenarios)}개")
    
    # 토큰 길이 범위
    total_tokens = [s['total_tokens'] for s in scenarios]
    print(f"📏 총 토큰 수 범위: {min(total_tokens)} ~ {max(total_tokens)} 토큰")


def main():
    """메인 실행 함수"""
    
    print("🚀 L4 + LLaMA 3.1 8B 특화 시나리오 생성기")
    print("=" * 60)
    
    # L4 전용 시나리오 생성
    print("📋 L4 + LLaMA 3.1 8B 시나리오 생성 중...")
    scenarios = generate_l4_llama31_scenarios()
    
    # CSV로 저장
    save_scenarios_to_csv(scenarios, 'l4_llama31_scenarios.csv')
    
    # 분석 결과 출력
    analyze_scenarios(scenarios)
    
    print(f"\n✅ L4 시나리오 생성 완료!")
    print(f"📁 파일 위치: /Users/anchovy-mac/Desktop/calculating/experiment/l4_llama31_scenarios.csv")
    
    # 사용법 안내
    print(f"\n💡 사용법:")
    print(f"cd /Users/anchovy-mac/Desktop/calculating/experiment")
    print(f"python3 run_prediction_experiment.py l4_llama31_scenarios.csv")


if __name__ == "__main__":
    main()