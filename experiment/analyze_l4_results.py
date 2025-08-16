"""
L4 + LLaMA 3.1 8B 성능 분석기

Input/Output 길이별 성능 패턴을 상세 분석하여
L4 GPU의 특성을 파악합니다.
"""

import pandas as pd
import numpy as np

def analyze_l4_performance():
    """L4 성능 결과 분석"""
    
    print("📊 L4 + LLaMA 3.1 8B 성능 분석")
    print("=" * 60)
    
    # 결과 파일 로드
    try:
        df = pd.read_csv('/Users/anchovy-mac/Desktop/calculating/experiment/l4_llama31_scenarios_predictions.csv')
        print(f"📖 {len(df)}개 예측 결과 로드됨")
    except FileNotFoundError:
        print("❌ 결과 파일을 찾을 수 없습니다. 먼저 실험을 실행하세요.")
        return
    
    print(f"\n🎯 전체 성능 요약:")
    print("-" * 40)
    print(f"평균 총 지연시간: {df['predicted_total_ms'].mean():.1f} ms")
    print(f"평균 처리량: {df['predicted_throughput_tokens_per_sec'].mean():.1f} tokens/sec")
    print(f"PREFILL 평균: {df['predicted_prefill_ms'].mean():.1f} ms")
    print(f"DECODE 평균: {df['predicted_decode_per_token_ms'].mean():.3f} ms/token")
    
    # Input 길이별 분석
    print(f"\n📥 Input 길이별 성능 분석:")
    print("-" * 40)
    input_analysis = df.groupby('input_length').agg({
        'predicted_total_ms': ['mean', 'min', 'max'],
        'predicted_prefill_ms': 'mean',
        'predicted_throughput_tokens_per_sec': 'mean'
    }).round(1)
    
    for input_len in sorted(df['input_length'].unique()):
        subset = df[df['input_length'] == input_len]
        avg_total = subset['predicted_total_ms'].mean()
        avg_prefill = subset['predicted_prefill_ms'].mean()
        avg_throughput = subset['predicted_throughput_tokens_per_sec'].mean()
        print(f"  • {input_len:4d} 토큰: {avg_total:6.1f}ms (PREFILL: {avg_prefill:6.1f}ms, {avg_throughput:5.1f} tok/sec)")
    
    # Output 길이별 분석
    print(f"\n📤 Output 길이별 성능 분석:")
    print("-" * 40)
    for output_len in sorted(df['output_length'].unique()):
        subset = df[df['output_length'] == output_len]
        avg_total = subset['predicted_total_ms'].mean()
        avg_decode_total = subset['predicted_total_decode_ms'].mean()
        avg_throughput = subset['predicted_throughput_tokens_per_sec'].mean()
        print(f"  • {output_len:4d} 토큰: {avg_total:6.1f}ms (DECODE: {avg_decode_total:6.1f}ms, {avg_throughput:5.1f} tok/sec)")
    
    # PREFILL vs DECODE 비율 분석
    print(f"\n⚖️  PREFILL vs DECODE 비율 분석:")
    print("-" * 40)
    df['prefill_ratio'] = df['predicted_prefill_ms'] / df['predicted_total_ms'] * 100
    df['decode_ratio'] = df['predicted_total_decode_ms'] / df['predicted_total_ms'] * 100
    
    print(f"평균 PREFILL 비율: {df['prefill_ratio'].mean():.1f}%")
    print(f"평균 DECODE 비율: {df['decode_ratio'].mean():.1f}%")
    
    # Input 길이에 따른 PREFILL 비율 변화
    print(f"\nInput 길이별 PREFILL 비율:")
    for input_len in sorted(df['input_length'].unique()):
        subset = df[df['input_length'] == input_len]
        avg_prefill_ratio = subset['prefill_ratio'].mean()
        print(f"  • {input_len:4d} 토큰: {avg_prefill_ratio:5.1f}%")
    
    # 성능 효율성 분석
    print(f"\n🚀 성능 효율성 분석:")
    print("-" * 40)
    
    # 최고/최저 성능 시나리오
    best_perf = df.loc[df['predicted_throughput_tokens_per_sec'].idxmax()]
    worst_perf = df.loc[df['predicted_throughput_tokens_per_sec'].idxmin()]
    
    print(f"최고 성능:")
    print(f"  • Input {best_perf['input_length']}, Output {best_perf['output_length']} → {best_perf['predicted_throughput_tokens_per_sec']:.1f} tok/sec")
    print(f"  • 총 시간: {best_perf['predicted_total_ms']:.1f}ms")
    
    print(f"최저 성능:")
    print(f"  • Input {worst_perf['input_length']}, Output {worst_perf['output_length']} → {worst_perf['predicted_throughput_tokens_per_sec']:.1f} tok/sec")
    print(f"  • 총 시간: {worst_perf['predicted_total_ms']:.1f}ms")
    
    # 사용 사례별 분석
    print(f"\n📝 사용 사례별 성능:")
    print("-" * 40)
    for use_case in df['use_case'].unique():
        subset = df[df['use_case'] == use_case]
        avg_latency = subset['predicted_total_ms'].mean()
        avg_throughput = subset['predicted_throughput_tokens_per_sec'].mean()
        count = len(subset)
        print(f"  • {use_case:18s}: {avg_latency:6.1f}ms, {avg_throughput:5.1f} tok/sec ({count}개)")
    
    # 스케일링 분석
    print(f"\n📈 스케일링 특성 분석:")
    print("-" * 40)
    
    # Input 길이 증가에 따른 PREFILL 시간 증가
    input_prefill = df.groupby('input_length')['predicted_prefill_ms'].mean()
    print("Input 스케일링 (PREFILL):")
    for i, (input_len, prefill_time) in enumerate(input_prefill.items()):
        if i > 0:
            prev_len, prev_time = list(input_prefill.items())[i-1]
            ratio = input_len / prev_len
            time_ratio = prefill_time / prev_time
            print(f"  • {prev_len}→{input_len} ({ratio:.1f}x): {prev_time:.1f}→{prefill_time:.1f}ms ({time_ratio:.2f}x)")
    
    # Output 길이 증가에 따른 DECODE 시간 증가
    output_decode = df.groupby('output_length')['predicted_total_decode_ms'].mean()
    print("\nOutput 스케일링 (DECODE):")
    for i, (output_len, decode_time) in enumerate(output_decode.items()):
        if i > 0:
            prev_len, prev_time = list(output_decode.items())[i-1]
            ratio = output_len / prev_len
            time_ratio = decode_time / prev_time if prev_time > 0 else float('inf')
            print(f"  • {prev_len}→{output_len} ({ratio:.1f}x): {prev_time:.1f}→{decode_time:.1f}ms ({time_ratio:.2f}x)")
    
    # 권장사항
    print(f"\n💡 L4 GPU 활용 권장사항:")
    print("-" * 40)
    
    # 가장 효율적인 구간 찾기
    efficient_scenarios = df[df['predicted_throughput_tokens_per_sec'] > df['predicted_throughput_tokens_per_sec'].quantile(0.75)]
    
    print("높은 효율성 구간:")
    for _, row in efficient_scenarios.iterrows():
        print(f"  • Input {row['input_length']:4d}, Output {row['output_length']:4d}: {row['predicted_throughput_tokens_per_sec']:5.1f} tok/sec")
    
    # 메모리 효율성 고려사항
    print(f"\n메모리 효율성 고려사항:")
    long_context = df[df['input_length'] >= 2048]
    if not long_context.empty:
        print(f"  • 긴 컨텍스트 (≥2048): 평균 {long_context['predicted_total_ms'].mean():.1f}ms")
        print(f"  • PREFILL 비중이 {long_context['prefill_ratio'].mean():.1f}%로 높음")
    
    print(f"\n✅ 분석 완료!")


if __name__ == "__main__":
    analyze_l4_performance()