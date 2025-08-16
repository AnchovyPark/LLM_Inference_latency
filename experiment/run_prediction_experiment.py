"""
LLM 지연시간 예측 실험 실행기

시나리오 CSV를 읽어서 우리 예측기로 예측을 수행하고 
결과를 CSV로 저장합니다.
"""

import csv
import os
import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict

# 상위 디렉토리의 예측기 import
sys.path.append('/Users/anchovy-mac/Desktop/calculating')
from llm_latency_predictor import LLMLatencyPredictor


def load_scenarios(csv_file: str) -> List[Dict]:
    """시나리오 CSV 파일 로드"""
    
    filepath = f'/Users/anchovy-mac/Desktop/calculating/experiment/{csv_file}'
    scenarios = []
    
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # 숫자 필드들을 int로 변환
            row['scenario_id'] = int(row['scenario_id'])
            row['input_length'] = int(row['input_length'])
            row['output_length'] = int(row['output_length'])
            row['batch_size'] = int(row['batch_size'])
            scenarios.append(row)
    
    return scenarios


def run_predictions(scenarios: List[Dict], predictor: LLMLatencyPredictor) -> List[Dict]:
    """시나리오들에 대해 예측 실행"""
    
    results = []
    total_scenarios = len(scenarios)
    
    print(f"🔮 {total_scenarios}개 시나리오에 대해 예측 실행 중...")
    print("-" * 60)
    
    for i, scenario in enumerate(scenarios, 1):
        try:
            # 예측 실행
            result = predictor.predict_latency(
                model_name=scenario['model'],
                gpu_name=scenario['gpu'], 
                input_length=scenario['input_length'],
                output_length=scenario['output_length'],
                batch_size=scenario['batch_size']
            )
            
            # 결과 저장
            prediction_result = {
                'scenario_id': scenario['scenario_id'],
                'gpu': scenario['gpu'],
                'model': scenario['model'],
                'input_length': scenario['input_length'],
                'output_length': scenario['output_length'],
                'batch_size': scenario['batch_size'],
                'use_case': scenario['use_case'],
                
                # 예측 결과들
                'predicted_prefill_ms': round(result.prefill_ms, 2),
                'predicted_decode_per_token_ms': round(result.decode_per_token_ms, 4),
                'predicted_total_decode_ms': round(result.total_decode_ms, 2),
                'predicted_total_ms': round(result.total_ms, 2),
                'predicted_throughput_tokens_per_sec': round(scenario['output_length'] / (result.total_ms / 1000), 2),
                
                # 세부 분석
                'predicted_linear_ms': round(result.linear_ms, 2),
                'predicted_attention_ms': round(result.attention_ms, 2),
                'predicted_mlp_ms': round(result.mlp_ms, 2),
                'predicted_norm_ms': round(result.norm_ms, 2),
                'predicted_other_ms': round(result.other_ms, 2),
                'predicted_memory_transfer_ms': round(result.memory_transfer_ms, 2),
                'predicted_compute_ms': round(result.compute_ms, 2),
                
                'prediction_timestamp': datetime.now().isoformat()
            }
            
            results.append(prediction_result)
            
            # 진행상황 출력
            if i % 10 == 0 or i == total_scenarios:
                print(f"✅ {i}/{total_scenarios} 완료 ({i/total_scenarios*100:.1f}%)")
                
        except Exception as e:
            print(f"❌ 시나리오 {scenario['scenario_id']} 실패: {e}")
            
            # 실패한 경우도 기록
            error_result = {
                'scenario_id': scenario['scenario_id'],
                'gpu': scenario['gpu'],
                'model': scenario['model'],
                'input_length': scenario['input_length'],
                'output_length': scenario['output_length'],
                'batch_size': scenario['batch_size'],
                'use_case': scenario['use_case'],
                'error': str(e),
                'prediction_timestamp': datetime.now().isoformat()
            }
            results.append(error_result)
    
    return results


def save_predictions_to_csv(predictions: List[Dict], filename: str):
    """예측 결과를 CSV로 저장"""
    
    filepath = f'/Users/anchovy-mac/Desktop/calculating/experiment/{filename}'
    
    if not predictions:
        print("❌ 저장할 예측 결과가 없습니다.")
        return
    
    # 필드명 정의 (에러가 있는 경우와 없는 경우 모두 고려)
    fieldnames = list(predictions[0].keys())
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"💾 예측 결과가 {filepath}에 저장되었습니다")


def analyze_predictions(predictions: List[Dict]):
    """예측 결과 분석"""
    
    # 성공/실패 통계
    successful = [p for p in predictions if 'error' not in p]
    failed = [p for p in predictions if 'error' in p]
    
    print(f"\n📈 예측 실행 결과 분석:")
    print("-" * 40)
    print(f"✅ 성공: {len(successful)}개")
    print(f"❌ 실패: {len(failed)}개")
    print(f"📊 성공률: {len(successful)/len(predictions)*100:.1f}%")
    
    if successful:
        df = pd.DataFrame(successful)
        
        print(f"\n⚡ 지연시간 통계 (ms):")
        print(f"  • 평균 총 지연시간: {df['predicted_total_ms'].mean():.1f} ms")
        print(f"  • 최소 총 지연시간: {df['predicted_total_ms'].min():.1f} ms")
        print(f"  • 최대 총 지연시간: {df['predicted_total_ms'].max():.1f} ms")
        
        print(f"\n🚀 처리량 통계 (tokens/sec):")
        print(f"  • 평균 처리량: {df['predicted_throughput_tokens_per_sec'].mean():.1f} tokens/sec")
        print(f"  • 최소 처리량: {df['predicted_throughput_tokens_per_sec'].min():.1f} tokens/sec")
        print(f"  • 최대 처리량: {df['predicted_throughput_tokens_per_sec'].max():.1f} tokens/sec")
        
        # GPU별 평균 지연시간
        print(f"\n🖥️  GPU별 평균 총 지연시간:")
        gpu_latency = df.groupby('gpu')['predicted_total_ms'].mean().sort_values()
        for gpu, latency in gpu_latency.items():
            print(f"  • {gpu}: {latency:.1f} ms")
    
    if failed:
        print(f"\n❌ 실패한 시나리오들:")
        for fail in failed[:5]:  # 처음 5개만 표시
            print(f"  • {fail['gpu']}-{fail['model']}: {fail.get('error', 'Unknown error')}")


def main():
    """메인 실험 실행 함수"""
    
    print("🧪 LLM 지연시간 예측 실험")
    print("=" * 50)
    
    # 예측기 초기화
    print("🔧 예측기 초기화 중...")
    predictor = LLMLatencyPredictor()
    
    # 명령행 인수 확인
    import sys
    if len(sys.argv) > 1:
        # 특정 시나리오 파일 처리
        scenario_file = sys.argv[1]
        if not scenario_file.endswith('.csv'):
            scenario_file += '.csv'
        
        output_file = scenario_file.replace('.csv', '_predictions.csv')
        scenario_files = [(scenario_file, output_file)]
    else:
        # 기본 시나리오 파일들
        scenario_files = [
            ('balanced_scenarios.csv', 'balanced_predictions.csv'),
            ('random_scenarios.csv', 'random_predictions.csv')
        ]
    
    for scenario_file, output_file in scenario_files:
        print(f"\n📋 {scenario_file} 처리 중...")
        
        try:
            # 시나리오 로드
            scenarios = load_scenarios(scenario_file)
            print(f"📖 {len(scenarios)}개 시나리오 로드됨")
            
            # 예측 실행
            predictions = run_predictions(scenarios, predictor)
            
            # 결과 저장
            save_predictions_to_csv(predictions, output_file)
            
            # 분석
            analyze_predictions(predictions)
            
        except FileNotFoundError:
            print(f"⚠️  {scenario_file}을 찾을 수 없습니다. 파일명을 확인하세요.")
        except Exception as e:
            print(f"❌ {scenario_file} 처리 중 오류 발생: {e}")
    
    print(f"\n🎉 실험 완료!")


if __name__ == "__main__":
    main()