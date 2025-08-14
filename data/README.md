# LLM Performance Delay Factor Analysis - Final Clean Data

이 폴더는 LLM 성능 분석을 위한 최종 정리된 데이터와 코드를 포함합니다.

## 📊 GPU 성능 데이터 (Delay Factors)

모든 delay factor는 **memory-bound 연산은 memory bandwidth 기준**, **compute-bound 연산은 Peak TFLOPS 기준**으로 정확히 계산되었습니다.

### GPU Delay Factor Files:
- `T4_DF_memory_corrected.json` - Tesla T4 성능 데이터
- `A10G_DF_memory_corrected.json` - NVIDIA A10G 성능 데이터  
- `L40s_DF_memory_corrected.json` - NVIDIA L40S 성능 데이터
- `A100_40G_DF_memory_corrected.json` - NVIDIA A100 40GB 성능 데이터
- `A100_80G_DF_memory_corrected.json` - NVIDIA A100 80GB 성능 데이터

### Delay Factor 범위 (정상적인 값들):
- **Compute-bound 연산** (Linear_GEMM, BMM, GQA, SwiGLU_MLP): 1.0x - 7.0x
- **Memory-bound 연산** (Elementwise_Add, RMS_Norm, Softmax): 1.0x - 17.0x

## 🤖 모델 Configuration Files

- `llama_3_2_1b_config.json` - LLaMA 3.2 1B 모델 설정
- `llama_3_8b_config.json` - LLaMA 3 8B 모델 설정  
- `llama_3_70b_config.json` - LLaMA 3 70B 모델 설정

## 🔧 핵심 코드 파일들

### `gpu_specs_database.py`
- GPU 하드웨어 스펙 데이터베이스
- 자동 GPU 감지 기능
- 지원 GPU: T4, A10G, L40S, A100, V100, RTX 4090 등

### `llm_latency_predictor.py`
- **메인 예측 시스템** 🎯
- 모델 + GPU + input/output length → latency 예측
- delay factor fine-tuning 지원
- PREFILL + DECODE 단계별 분석

## 🚀 사용법

### 1. 기본 latency 예측:
```python
from llm_latency_predictor import LLMLatencyPredictor

predictor = LLMLatencyPredictor()

# LLaMA 3.2 1B + T4 + 512 input + 100 output tokens
result = predictor.predict_latency('LLaMA_3.2_1B', 'T4', 512, 100)
print(f"Total latency: {result.total_ms:.1f}ms")
```

### 2. Delay factor 조정:
```python
# Linear 연산을 20% 빠르게, Attention을 10% 느리게 조정
predictor.fine_tune_delay_factors({
    'T4_Linear_GEMM': 0.8,
    'T4_BMM': 1.1
})
```

### 3. 지원되는 모델:
- `'LLaMA_3.2_1B'` - 2048 hidden, 16 layers
- `'LLaMA_3_8B'` - 4096 hidden, 32 layers  
- `'LLaMA_3_70B'` - 8192 hidden, 80 layers

### 4. 지원되는 GPU:
- `'T4'`, `'A10G'`, `'L40S'`, `'A100'`

## 📈 성능 특성 요약

### GPU 성능 순위 (전체 평균 delay factor 기준):
1. **L40S**: 가장 효율적 (1.0x-6.0x 범위)
2. **A100**: 고성능 범용 GPU (1.5x-17x 범위)  
3. **A10G**: 중급 성능 (2.0x-7.0x 범위)
4. **T4**: 경제적 옵션 (2.0x-7.0x 범위)

### 연산별 특성:
- **Linear_GEMM**: 가장 최적화된 연산 (1-4x delay)
- **SwiGLU_MLP**: 매우 효율적 (1-2x delay)
- **BMM/Attention**: 중간 수준 (3-7x delay)  
- **Memory-bound 연산**: 상대적으로 높은 delay (2-17x)

## ⚠️ 주의사항

1. **실제 환경에서는 추가 오버헤드 발생 가능**: 메모리 전송, 커널 런치 등
2. **Batch size 1 기준**: 더 큰 batch size에서는 효율성 향상 가능
3. **FP16 기준 측정**: FP32나 다른 precision에서는 다른 결과 가능
4. **GPU별 최적화 차이**: 드라이버 버전, CUDA 버전에 따라 차이 가능

## 🎯 권장 사용 시나리오

- **개발/테스팅**: T4 (저비용)
- **프로덕션 추론**: A10G, L40S (성능/비용 균형)  
- **대용량/연구**: A100 (최고 성능)
- **모델 크기별**: 1B→T4, 8B→A10G/L40S, 70B→A100

---
📅 마지막 업데이트: 2024-08-14
🔬 측정 환경: FP16, Single GPU, Batch Size 1
✅ 모든 delay factor 검증 완료