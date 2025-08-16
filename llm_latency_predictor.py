"""
LLM Latency Prediction System

End-to-end system that predicts LLM inference latency based on:
- Model architecture (LLaMA 3.2 1B, 8B, 70B, etc.)
- GPU hardware (T4, A10G, L40S, etc.) 
- Input sequence length
- Output sequence length (for generation)

Supports both PREFILL and DECODE phase predictions with realistic delay factors.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from gpu_specs_database import GPUSpecsDatabase, GPUSpecs


@dataclass
class ModelConfig:
    """LLM model configuration."""
    name: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_layers: int
    vocab_size: int


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model: ModelConfig
    gpu: GPUSpecs
    input_length: int
    output_length: int
    batch_size: int = 1
    
    
@dataclass 
class LatencyBreakdown:
    """Detailed latency breakdown."""
    prefill_ms: float
    decode_per_token_ms: float
    total_decode_ms: float
    total_ms: float
    
    # Detailed operator breakdown
    linear_ms: float
    attention_ms: float  
    mlp_ms: float
    norm_ms: float
    other_ms: float
    
    # Memory analysis
    memory_transfer_ms: float
    compute_ms: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class DelayFactorManager:
    """Manages and allows fine-tuning of delay factors."""
    
    def __init__(self):
        self.delay_factors = {}
        self.custom_adjustments = {}
        
    def load_delay_factors(self, gpu_name: str) -> None:
        """Load delay factors for a specific GPU."""
        
        # GPU name mapping for delay factor files
        gpu_file_mapping = {
            'A100': 'A100_40G',
            'A100-SXM4-40GB': 'A100_40G', 
            'A100-SXM4-80GB': 'A100_80G',
            'Tesla T4': 'T4',
            'T4': 'T4',
            'L4': 'L4',  # NVIDIA L4
            'NVIDIA L4': 'L4',
            'A10G': 'A10G',
            'L40S': 'L40s',  # Note: L40s with lowercase 's' in filename
            'L40s': 'L40s'
        }
        
        # Map GPU name to delay factor filename
        file_gpu_name = gpu_file_mapping.get(gpu_name, gpu_name)
        
        # Try to load memory-corrected version first, then regular version, then basic DF file
        corrected_file = f'/Users/anchovy-mac/Desktop/calculating/data/{file_gpu_name}_DF_memory_corrected.json'
        fallback_file = f'/Users/anchovy-mac/Desktop/calculating/data/{file_gpu_name}_DF_corrected.json'
        basic_file = f'/Users/anchovy-mac/Desktop/calculating/data/{file_gpu_name}_DF.json'
        
        if os.path.exists(corrected_file):
            file_to_load = corrected_file
        elif os.path.exists(fallback_file):
            file_to_load = fallback_file
        elif os.path.exists(basic_file):
            file_to_load = basic_file
        else:
            raise FileNotFoundError(f"No delay factor file found for GPU: {gpu_name} (tried {file_gpu_name})")
        
        with open(file_to_load, 'r') as f:
            data = json.load(f)
            
        self.delay_factors[gpu_name] = data['delay_factors']
        print(f"📊 Loaded delay factors for {gpu_name} from {os.path.basename(file_to_load)}")
    
    def get_delay_factor(self, gpu_name: str, operator: str) -> float:
        """Get delay factor for specific GPU and operator with custom adjustments."""
        
        if gpu_name not in self.delay_factors:
            self.load_delay_factors(gpu_name)
        
        base_factor = self.delay_factors[gpu_name][operator]['avg_delay_factor']
        
        # Apply custom adjustment if exists
        adjustment_key = f"{gpu_name}_{operator}"
        if adjustment_key in self.custom_adjustments:
            adjusted_factor = base_factor * self.custom_adjustments[adjustment_key]
            print(f"🔧 Using custom adjustment for {adjustment_key}: {base_factor:.2f} → {adjusted_factor:.2f}")
            return adjusted_factor
        
        return base_factor
    
    def set_custom_adjustment(self, gpu_name: str, operator: str, multiplier: float) -> None:
        """Set custom multiplier for delay factor fine-tuning."""
        adjustment_key = f"{gpu_name}_{operator}"
        self.custom_adjustments[adjustment_key] = multiplier
        print(f"🎛️  Set custom adjustment: {adjustment_key} × {multiplier}")
    
    def reset_adjustments(self) -> None:
        """Reset all custom adjustments."""
        self.custom_adjustments.clear()
        print("🔄 Reset all custom delay factor adjustments")
    
    def save_adjustments(self, file_path: str) -> None:
        """Save custom adjustments to file."""
        with open(file_path, 'w') as f:
            json.dump(self.custom_adjustments, f, indent=2)
        print(f"💾 Saved custom adjustments to {file_path}")
    
    def load_adjustments(self, file_path: str) -> None:
        """Load custom adjustments from file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.custom_adjustments = json.load(f)
            print(f"📂 Loaded custom adjustments from {file_path}")
        else:
            print(f"⚠️  Adjustment file not found: {file_path}")


class LLMLatencyPredictor:
    """Main LLM latency prediction system."""
    
    def __init__(self):
        self.gpu_db = GPUSpecsDatabase()
        self.delay_factor_manager = DelayFactorManager()
        
        # Predefined model configurations
        self.model_configs = {
            'LLaMA_3.2_1B': ModelConfig(
                name='LLaMA_3.2_1B',
                hidden_size=2048,
                intermediate_size=5632, 
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=64,
                num_layers=16,
                vocab_size=128256
            ),
            'LLaMA_3_8B': ModelConfig(
                name='LLaMA_3_8B',
                hidden_size=4096,
                intermediate_size=14336,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                num_layers=32,
                vocab_size=128256
            ),
            'LLaMA_3_70B': ModelConfig(
                name='LLaMA_3_70B',
                hidden_size=8192,
                intermediate_size=28672,
                num_attention_heads=64,
                num_key_value_heads=8,
                head_dim=128,
                num_layers=80,
                vocab_size=128256
            ),
            'LLaMA_3.1_8B': ModelConfig(
                name='LLaMA_3.1_8B',
                hidden_size=4096,
                intermediate_size=14336,
                num_attention_heads=32,
                num_key_value_heads=8,
                head_dim=128,
                num_layers=32,
                vocab_size=128256
            )
        }
    
    def predict_latency(self, model_name: str, gpu_name: str, input_length: int, 
                       output_length: int, batch_size: int = 1) -> LatencyBreakdown:
        """
        Predict end-to-end latency for LLM inference.
        
        Args:
            model_name: Model identifier (e.g., 'LLaMA_3.2_1B')
            gpu_name: GPU identifier (e.g., 'T4', 'A10G', 'L40S')
            input_length: Input sequence length 
            output_length: Number of tokens to generate
            batch_size: Batch size (default 1)
            
        Returns:
            Detailed latency breakdown
        """
        
        # Get configurations
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")
        
        model = self.model_configs[model_name]
        gpu_specs = self.gpu_db.get_specs_by_name(gpu_name)
        
        if gpu_specs is None:
            raise ValueError(f"Unknown GPU: {gpu_name}")
        
        config = InferenceConfig(
            model=model,
            gpu=gpu_specs,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size
        )
        
        # Calculate PREFILL phase (process input tokens)
        prefill_latency = self._calculate_prefill_latency(config)
        
        # Calculate DECODE phase (generate output tokens)
        decode_per_token = self._calculate_decode_per_token_latency(config)
        total_decode = decode_per_token * output_length
        
        # Total latency
        total_latency = prefill_latency.total_ms + total_decode
        
        # Combine results
        return LatencyBreakdown(
            prefill_ms=prefill_latency.total_ms,
            decode_per_token_ms=decode_per_token,
            total_decode_ms=total_decode,
            total_ms=total_latency,
            linear_ms=prefill_latency.linear_ms + (decode_per_token * output_length * 0.6),  # Approximation
            attention_ms=prefill_latency.attention_ms + (decode_per_token * output_length * 0.3),
            mlp_ms=prefill_latency.mlp_ms + (decode_per_token * output_length * 0.1),
            norm_ms=prefill_latency.norm_ms,
            other_ms=prefill_latency.other_ms,
            memory_transfer_ms=prefill_latency.memory_transfer_ms + (total_decode * 0.2),
            compute_ms=prefill_latency.compute_ms + (total_decode * 0.8)
        )
    
    def _calculate_prefill_latency(self, config: InferenceConfig) -> LatencyBreakdown:
        """Calculate PREFILL phase latency (process input sequence)."""
        
        model = config.model
        batch_size = config.batch_size
        seq_len = config.input_length
        
        # Linear projections (major contributor in PREFILL)
        linear_ops = []
        
        for layer in range(model.num_layers):
            # Attention projections: Q, K, V, O
            q_ops = batch_size * seq_len * model.hidden_size * model.hidden_size * 2
            kv_ops = batch_size * seq_len * model.hidden_size * (model.num_key_value_heads * model.head_dim) * 2 * 2  # K and V
            o_ops = batch_size * seq_len * model.hidden_size * model.hidden_size * 2
            
            # MLP projections: Gate, Up, Down  
            gate_ops = batch_size * seq_len * model.hidden_size * model.intermediate_size * 2
            up_ops = batch_size * seq_len * model.hidden_size * model.intermediate_size * 2
            down_ops = batch_size * seq_len * model.intermediate_size * model.hidden_size * 2
            
            linear_ops.extend([q_ops, kv_ops, o_ops, gate_ops, up_ops, down_ops])
        
        # Attention computation (BMM operations)
        attention_ops = []
        for layer in range(model.num_layers):
            # Q @ K^T
            qk_ops = batch_size * model.num_attention_heads * seq_len * seq_len * model.head_dim * 2
            # Attn @ V  
            av_ops = batch_size * model.num_attention_heads * seq_len * seq_len * model.head_dim * 2
            attention_ops.extend([qk_ops, av_ops])
        
        # Get delay factors
        linear_delay = self.delay_factor_manager.get_delay_factor(config.gpu.name, 'Linear_GEMM')
        attention_delay = self.delay_factor_manager.get_delay_factor(config.gpu.name, 'BMM')
        mlp_delay = self.delay_factor_manager.get_delay_factor(config.gpu.name, 'SwiGLU_MLP') 
        norm_delay = self.delay_factor_manager.get_delay_factor(config.gpu.name, 'RMS_Norm')
        
        # Calculate latencies
        peak_tflops = config.gpu.peak_fp16_tflops * 1e12
        
        linear_theoretical_time = sum(linear_ops) / peak_tflops
        linear_actual_time = linear_theoretical_time * linear_delay
        
        attention_theoretical_time = sum(attention_ops) / peak_tflops  
        attention_actual_time = attention_theoretical_time * attention_delay
        
        # Normalization operations (memory-bound)
        norm_elements = batch_size * seq_len * model.hidden_size * model.num_layers * 2  # Pre-attn and pre-MLP
        norm_bytes = norm_elements * 2 * 2  # fp16, read+write
        norm_theoretical_time = norm_bytes / (config.gpu.memory_bandwidth_gb_s * 1e9)
        norm_actual_time = norm_theoretical_time * norm_delay
        
        # Other operations (embeddings, etc.)
        other_time = linear_actual_time * 0.1  # Estimate 10% overhead
        
        total_time = linear_actual_time + attention_actual_time + norm_actual_time + other_time
        
        return LatencyBreakdown(
            prefill_ms=0,  # Will be set to total_ms
            decode_per_token_ms=0,
            total_decode_ms=0,
            total_ms=total_time * 1000,  # Convert to ms
            linear_ms=linear_actual_time * 1000,
            attention_ms=attention_actual_time * 1000,
            mlp_ms=0,  # Included in linear for simplicity
            norm_ms=norm_actual_time * 1000,
            other_ms=other_time * 1000,
            memory_transfer_ms=(norm_actual_time + other_time) * 1000,
            compute_ms=(linear_actual_time + attention_actual_time) * 1000
        )
    
    def _calculate_decode_per_token_latency(self, config: InferenceConfig) -> float:
        """Calculate per-token latency in DECODE phase."""
        
        # In decode phase, we process 1 token at a time but have growing KV cache
        model = config.model
        batch_size = config.batch_size
        kv_seq_len = config.input_length  # KV cache length (grows each step)
        
        # Per-token operations (much smaller than PREFILL)
        per_token_ops = 0
        
        for layer in range(model.num_layers):
            # Linear projections for 1 new token
            q_ops = batch_size * 1 * model.hidden_size * model.hidden_size * 2
            kv_ops = batch_size * 1 * model.hidden_size * (model.num_key_value_heads * model.head_dim) * 2 * 2
            o_ops = batch_size * 1 * model.hidden_size * model.hidden_size * 2
            
            # MLP for 1 new token
            mlp_ops = batch_size * 1 * (model.hidden_size * model.intermediate_size * 2 * 2 + 
                                        model.intermediate_size * model.hidden_size * 2)
            
            # Attention with KV cache (1 new token attends to all previous tokens)
            attn_ops = batch_size * model.num_attention_heads * 1 * kv_seq_len * model.head_dim * 2 * 2
            
            per_token_ops += q_ops + kv_ops + o_ops + mlp_ops + attn_ops
        
        # Apply delay factors
        linear_delay = self.delay_factor_manager.get_delay_factor(config.gpu.name, 'Linear_GEMM')
        
        theoretical_time = per_token_ops / (config.gpu.peak_fp16_tflops * 1e12)
        actual_time = theoretical_time * linear_delay  # Use linear delay as representative
        
        return actual_time * 1000  # Convert to ms
    
    def fine_tune_delay_factors(self, adjustments: Dict[str, float]) -> None:
        """Fine-tune delay factors with custom multipliers.
        
        Args:
            adjustments: Dict of {"{gpu}_{operator}": multiplier} 
        """
        for key, multiplier in adjustments.items():
            if '_' in key:
                gpu, operator = key.rsplit('_', 1)
                self.delay_factor_manager.set_custom_adjustment(gpu, operator, multiplier)
    
    def benchmark_prediction_accuracy(self, real_measurements: List[Dict]) -> Dict[str, float]:
        """Compare predictions against real measurements for accuracy assessment."""
        
        errors = []
        
        for measurement in real_measurements:
            predicted = self.predict_latency(
                measurement['model'], measurement['gpu'],
                measurement['input_length'], measurement['output_length']
            )
            
            actual = measurement['actual_ms']
            predicted_ms = predicted.total_ms
            
            error = abs(predicted_ms - actual) / actual * 100  # Percentage error
            errors.append(error)
        
        return {
            'mean_error_percent': sum(errors) / len(errors),
            'max_error_percent': max(errors),
            'min_error_percent': min(errors),
            'predictions_count': len(errors)
        }


def get_supported_info():
    """Get information about supported GPUs and models."""
    import glob
    
    # Detect available delay factor files
    df_files = glob.glob('/Users/anchovy-mac/Desktop/calculating/data/*_DF_memory_corrected.json')
    available_gpus = []
    for file in df_files:
        gpu_name = os.path.basename(file).replace('_DF_memory_corrected.json', '')
        available_gpus.append(gpu_name)
    
    # Map to user-friendly names
    gpu_mapping = {
        'T4': 'T4 (Tesla T4)',
        'A10G': 'A10G',
        'L40s': 'L40S',
        'A100_40G': 'A100 (40GB)',
        'A100_80G': 'A100_80G (80GB)'
    }
    
    friendly_gpus = [gpu_mapping.get(gpu, gpu) for gpu in available_gpus]
    
    # Available models
    available_models = ['LLaMA_3.2_1B', 'LLaMA_3_8B', 'LLaMA_3.1_8B', 'LLaMA_3_70B']
    
    return friendly_gpus, available_models


def print_help_with_info():
    """Print enhanced help information."""
    gpus, models = get_supported_info()
    
    print("🚀 LLM LATENCY PREDICTOR - HELP")
    print("=" * 60)
    print()
    print("📋 USAGE:")
    print("  python3 llm_latency_predictor.py [options]")
    print()
    print("🔧 OPTIONS:")
    print("  --model MODEL    Model to predict (required)")
    print("  --gpu GPU        GPU to use (required)")
    print("  --input INPUT    Input tokens (required)")
    print("  --output OUTPUT  Output tokens (required)")
    print("  --batch BATCH    Batch size (default: 1)")
    print("  --demo           Run demo mode")
    print("  -h, --help       Show this help")
    print()
    print("🖥️  SUPPORTED GPUs (with delay factors):")
    for gpu in gpus:
        print(f"  • {gpu}")
    print()
    print("🤖 SUPPORTED MODELS:")
    for model in models:
        print(f"  • {model}")
    print()
    print("💡 USAGE EXAMPLES:")
    print("  # Basic prediction")
    print('  python3 llm_latency_predictor.py --model "8b" --gpu "t4" --input 512 --output 100')
    print()
    print("  # A100 40GB (default)")
    print('  python3 llm_latency_predictor.py --model "llama_3_8b" --gpu "a100" --input 1024 --output 200')
    print()
    print("  # A100 80GB")
    print('  python3 llm_latency_predictor.py --model "70b" --gpu "a100_80g" --input 2048 --output 500')
    print()
    print("  # Demo mode (multiple test cases)")
    print('  python3 llm_latency_predictor.py --demo')
    print()
    print("📝 MODEL ALIASES:")
    print("  • 1b, llama_3_2_1b, llama_3.2_1b → LLaMA_3.2_1B")
    print("  • 8b, llama_3_8b, llama_3.8b → LLaMA_3_8B")
    print("  • 3.1_8b, llama_3.1_8b, llama_3_1_8b → LLaMA_3.1_8B")
    print("  • 70b, llama_3_70b, llama_3.70b → LLaMA_3_70B")
    print()
    print("🖥️  GPU ALIASES:")
    print("  • t4 → Tesla T4")
    print("  • a10g, a10 → A10G")
    print("  • l40s, l40 → L40S")
    print("  • a100 → A100 (40GB)")
    print("  • a100_80g → A100 (80GB)")


def main():
    """Demo of the LLM latency prediction system."""
    
    print("🚀 LLM LATENCY PREDICTION SYSTEM")
    print("=" * 80)
    
    predictor = LLMLatencyPredictor()
    
    # Example predictions
    test_cases = [
        {'model': 'LLaMA_3.2_1B', 'gpu': 'T4', 'input': 512, 'output': 100},
        {'model': 'LLaMA_3.2_1B', 'gpu': 'A10G', 'input': 512, 'output': 100},
        {'model': 'LLaMA_3.2_1B', 'gpu': 'L40S', 'input': 512, 'output': 100},
        {'model': 'LLaMA_3_8B', 'gpu': 'L40S', 'input': 1024, 'output': 200},
        {'model': 'LLaMA_3_70B', 'gpu': 'L40S', 'input': 2048, 'output': 500},
    ]
    
    print("📊 LATENCY PREDICTIONS:")
    print("-" * 80)
    print(f"{'Model':<15} {'GPU':<6} {'Input':<6} {'Output':<7} {'Total(ms)':<10} {'PREFILL':<10} {'DECODE/tok':<12}")
    print("-" * 80)
    
    for case in test_cases:
        try:
            result = predictor.predict_latency(
                case['model'], case['gpu'], 
                case['input'], case['output']
            )
            
            print(f"{case['model']:<15} {case['gpu']:<6} {case['input']:<6} "
                  f"{case['output']:<7} {result.total_ms:<10.1f} "
                  f"{result.prefill_ms:<10.1f} {result.decode_per_token_ms:<12.2f}")
                  
        except Exception as e:
            print(f"❌ Error: {case} - {e}")
    
    # Demo fine-tuning
    print(f"\n🎛️  DELAY FACTOR FINE-TUNING DEMO:")
    print("-" * 80)
    
    # Show original prediction
    original = predictor.predict_latency('LLaMA_3.2_1B', 'T4', 512, 100)
    print(f"Original T4 prediction: {original.total_ms:.1f}ms")
    
    # Apply fine-tuning
    adjustments = {
        'T4_Linear_GEMM': 0.8,  # Make linear ops 20% faster
        'T4_BMM': 1.2           # Make attention 20% slower
    }
    predictor.fine_tune_delay_factors(adjustments)
    
    # Show tuned prediction  
    tuned = predictor.predict_latency('LLaMA_3.2_1B', 'T4', 512, 100)
    print(f"Fine-tuned T4 prediction: {tuned.total_ms:.1f}ms ({tuned.total_ms - original.total_ms:+.1f}ms)")
    
    print(f"\n💡 USAGE EXAMPLES:")
    print("predictor = LLMLatencyPredictor()")
    print("result = predictor.predict_latency('LLaMA_3.2_1B', 'T4', 512, 100)")
    print("predictor.fine_tune_delay_factors({'T4_Linear_GEMM': 0.9})")


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="LLM Latency Predictor",
        add_help=False)  # Disable default help
    
    parser.add_argument('--model', type=str, 
                        help='Model to use for prediction (e.g., llama_3_8b, 8b, LLaMA_3_8B)')
    parser.add_argument('--gpu', type=str,
                        help='GPU to use for prediction (e.g., T4, A10G, L40S, A100_40G, A100_80G)')
    parser.add_argument('--input', type=int,
                        help='Input sequence length (number of tokens)')
    parser.add_argument('--output', type=int,
                        help='Output sequence length (number of tokens to generate)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo mode with multiple test cases')
    parser.add_argument('-h', '--help', action='store_true',
                        help='Show this help message')
    
    args = parser.parse_args()
    
    # Handle help first
    if args.help:
        print_help_with_info()
        sys.exit(0)
    
    if args.demo:
        # Run original demo
        main()
    elif args.model and args.gpu and args.input is not None and args.output is not None:
        # CLI prediction mode
        print("🚀 LLM Latency Predictor")
        print("=" * 50)
        
        predictor = LLMLatencyPredictor()
        
        # Map user-friendly names to internal model names (case insensitive)
        model_mapping = {
            'llama_3_2_1b': 'LLaMA_3.2_1B',
            'llama_3.2_1b': 'LLaMA_3.2_1B',
            'llama_3_8b': 'LLaMA_3_8B',
            'llama_3.8b': 'LLaMA_3_8B',
            'llama_3.1_8b': 'LLaMA_3.1_8B',
            'llama_3_1_8b': 'LLaMA_3.1_8B',
            '3.1_8b': 'LLaMA_3.1_8B',
            'llama_3_70b': 'LLaMA_3_70B',
            'llama_3.70b': 'LLaMA_3_70B',
            '1b': 'LLaMA_3.2_1B',
            '8b': 'LLaMA_3_8B',
            '70b': 'LLaMA_3_70B'
        }
        
        # Map user-friendly GPU names to internal names (case insensitive)
        gpu_mapping = {
            't4': 'Tesla T4',
            'a10g': 'A10G', 
            'a10': 'A10G',
            'l40s': 'L40S',
            'l40': 'L40S',
            'a100_40g': 'A100-SXM4-40GB',
            'a100_80g': 'A100-SXM4-80GB',
            'a100': 'A100'  # Default A100
        }
        
        # Normalize input names
        model_key = args.model.lower()
        gpu_key = args.gpu.lower()
        
        # Map to internal names
        internal_model = model_mapping.get(model_key, args.model)
        internal_gpu = gpu_mapping.get(gpu_key, args.gpu)
        
        try:
            result = predictor.predict_latency(
                model_name=internal_model,
                gpu_name=internal_gpu,
                input_length=args.input,
                output_length=args.output,
                batch_size=args.batch
            )
            
            print(f"Model: {args.model}")
            print(f"GPU: {args.gpu}")
            print(f"Input tokens: {args.input}")
            print(f"Output tokens: {args.output}")
            print(f"Batch size: {args.batch}")
            print("-" * 30)
            print(f"PREFILL time: {result.prefill_ms:.1f} ms")
            print(f"DECODE per token: {result.decode_per_token_ms:.2f} ms")
            print(f"Total DECODE time: {result.total_decode_ms:.1f} ms")
            print(f"TOTAL time: {result.total_ms:.1f} ms")
            print(f"Throughput: {args.output / (result.total_ms / 1000):.1f} tokens/sec")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        print_help_with_info()
        sys.exit(1)