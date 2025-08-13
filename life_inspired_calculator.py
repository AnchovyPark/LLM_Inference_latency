"""
LIFE-Inspired Operator-Level Calculator
Based on "Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling"
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    """Supported data types with their byte sizes."""
    FP32 = 4
    FP16 = 2
    BF16 = 2
    INT8 = 1
    INT4 = 0.5


@dataclass
class OperatorConfig:
    """Configuration for individual operators."""
    shape_a: Tuple[int, ...]
    shape_b: Optional[Tuple[int, ...]] = None
    dtype_a: DataType = DataType.FP16
    dtype_b: DataType = DataType.FP16
    dtype_out: DataType = DataType.FP16
    bias: bool = False
    mode: str = "eager"  # eager, fused


class FoundationalOperators:
    """Foundational operators as defined in LIFE paper Table 1."""
    
    @staticmethod
    def linear_gemm_bias(config: OperatorConfig) -> Tuple[int, int, int]:
        """
        Linear layer with GEMM + Bias operation.
        Returns: (compute_ops, mem_read_bytes, mem_write_bytes)
        """
        m, k = config.shape_a
        k2, n = config.shape_b
        assert k == k2, f"Matrix dimensions mismatch: {k} != {k2}"
        
        # Compute: 2*m*k*n operations (GEMM) + m*n (bias if enabled)
        compute_ops = 2 * m * k * n
        if config.bias:
            compute_ops += m * n
        
        # Memory Read: Input A + Weights B + Bias (if enabled)
        mem_read = (m * k) * config.dtype_a.value  # Input
        mem_read += (k * n) * config.dtype_b.value  # Weights
        if config.bias:
            mem_read += n * config.dtype_a.value  # Bias
        
        # Memory Write: Output
        mem_write = (m * n) * config.dtype_out.value
        
        return compute_ops, mem_read, mem_write
    
    @staticmethod
    def quantized_linear(config: OperatorConfig, grp_size: int = 128) -> Tuple[int, int, int]:
        """
        Quantized linear layer with dequantization.
        """
        m, k = config.shape_a
        k2, n = config.shape_b
        
        # Base GEMM operations
        compute_ops, mem_read, mem_write = FoundationalOperators.linear_gemm_bias(config)
        
        # Additional dequantization operations
        if config.dtype_b == DataType.INT4:
            compute_ops += (k * n) * 2  # Dequantization: shift + scale
            
            # Additional memory for quantization parameters
            num_groups = (k // grp_size) * n
            mem_read += num_groups * config.dtype_a.value  # scale
            mem_read += num_groups * config.dtype_b.value  # zero point
        
        return compute_ops, mem_read, mem_write
    
    @staticmethod
    def bmm(config: OperatorConfig) -> Tuple[int, int, int]:
        """
        Batch Matrix Multiplication.
        """
        b, m, k = config.shape_a
        b2, k2, n = config.shape_b
        assert b == b2 and k == k2, f"BMM shape mismatch"
        
        # Compute: 2*b*m*k*n - b*m*n
        compute_ops = 2 * b * m * k * n - b * m * n
        
        # Memory
        mem_read = (b * m * k + b * k * n) * config.dtype_a.value
        mem_write = (b * m * n) * config.dtype_out.value
        
        return compute_ops, mem_read, mem_write
    
    @staticmethod
    def elementwise(config: OperatorConfig) -> Tuple[int, int, int]:
        """
        Element-wise operations (add, multiply, etc.).
        """
        if len(config.shape_a) == 2:
            m, n = config.shape_a
        else:
            m = 1
            n = config.shape_a[0]
        
        compute_ops = m * n
        mem_read = 2 * m * n * config.dtype_a.value  # Two inputs
        mem_write = m * n * config.dtype_out.value
        
        return compute_ops, mem_read, mem_write
    
    @staticmethod
    def embedding(vocab_size: int, hidden_size: int, dtype: DataType) -> Tuple[int, int, int]:
        """
        Embedding lookup operation.
        """
        compute_ops = 1  # Lookup operation
        mem_read = vocab_size * hidden_size * dtype.value + hidden_size * dtype.value
        mem_write = hidden_size * dtype.value
        
        return compute_ops, mem_read, mem_write


class DerivedOperators:
    """Derived operators built from foundational operators."""
    
    @staticmethod
    def rms_norm(seq_len: int, hidden_dim: int, dtype: DataType) -> Tuple[int, int, int]:
        """
        RMS Normalization as used in Llama.
        RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
        """
        # Operations: square, mean, rsqrt, multiply by input, multiply by weight
        square_ops = seq_len * hidden_dim
        mean_ops = seq_len * hidden_dim  # Mean along last dimension
        rsqrt_ops = seq_len
        normalize_ops = seq_len * hidden_dim
        weight_ops = seq_len * hidden_dim
        
        compute_ops = square_ops + mean_ops + rsqrt_ops + normalize_ops + weight_ops
        
        # Memory: input + weight parameter
        mem_read = seq_len * hidden_dim * dtype.value + hidden_dim * dtype.value
        mem_write = seq_len * hidden_dim * dtype.value
        
        return compute_ops, mem_read, mem_write
    
    @staticmethod
    def rope(seq_len: int, num_heads: int, head_dim: int, dtype: DataType) -> Tuple[int, int, int]:
        """
        Rotary Position Embedding (RoPE).
        """
        # Approximate: cos/sin computation + rotation
        compute_ops = seq_len * num_heads * head_dim * 4
        
        # Memory for sin/cos tables and input/output
        mem_read = seq_len * num_heads * head_dim * dtype.value
        mem_write = seq_len * num_heads * head_dim * dtype.value
        
        return compute_ops, mem_read, mem_write
    
    @staticmethod
    def softmax(seq_len: int, num_heads: int, dtype: DataType) -> Tuple[int, int, int]:
        """
        Softmax operation in attention.
        """
        # exp, sum, divide operations
        compute_ops = 3 * num_heads * seq_len * seq_len
        
        mem_read = num_heads * seq_len * seq_len * dtype.value
        mem_write = num_heads * seq_len * seq_len * dtype.value
        
        return compute_ops, mem_read, mem_write
    
    @staticmethod
    def mha_attention(seq_len: int, hidden_dim: int, num_heads: int, dtype: DataType) -> Dict[str, Tuple[int, int, int]]:
        """
        Multi-Head Attention with detailed breakdown.
        """
        head_dim = hidden_dim // num_heads
        
        results = {}
        
        # Q, K, V projections
        q_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, hidden_dim), dtype, dtype)
        k_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, hidden_dim), dtype, dtype)
        v_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, hidden_dim), dtype, dtype)
        
        results['q_projection'] = FoundationalOperators.linear_gemm_bias(q_config)
        results['k_projection'] = FoundationalOperators.linear_gemm_bias(k_config)
        results['v_projection'] = FoundationalOperators.linear_gemm_bias(v_config)
        
        # RoPE on Q and K
        results['rope_q'] = DerivedOperators.rope(seq_len, num_heads, head_dim, dtype)
        results['rope_k'] = DerivedOperators.rope(seq_len, num_heads, head_dim, dtype)
        
        # Attention scores: Q @ K^T
        qk_config = OperatorConfig((num_heads, seq_len, head_dim), (num_heads, head_dim, seq_len), dtype, dtype)
        results['attention_scores'] = FoundationalOperators.bmm(qk_config)
        
        # Softmax
        results['softmax'] = DerivedOperators.softmax(seq_len, num_heads, dtype)
        
        # Attention @ V
        av_config = OperatorConfig((num_heads, seq_len, seq_len), (num_heads, seq_len, head_dim), dtype, dtype)
        results['attention_output'] = FoundationalOperators.bmm(av_config)
        
        # Output projection
        o_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, hidden_dim), dtype, dtype)
        results['output_projection'] = FoundationalOperators.linear_gemm_bias(o_config)
        
        return results
    
    @staticmethod
    def gqa_attention(seq_len: int, hidden_dim: int, num_q_heads: int, num_kv_heads: int, dtype: DataType) -> Dict[str, Tuple[int, int, int]]:
        """
        Grouped Query Attention (GQA) with reduced K/V heads.
        """
        head_dim = hidden_dim // num_q_heads
        kv_dim = num_kv_heads * head_dim
        
        results = {}
        
        # Q projection (full size)
        q_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, hidden_dim), dtype, dtype)
        results['q_projection'] = FoundationalOperators.linear_gemm_bias(q_config)
        
        # K, V projections (reduced size)
        k_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, kv_dim), dtype, dtype)
        v_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, kv_dim), dtype, dtype)
        results['k_projection'] = FoundationalOperators.linear_gemm_bias(k_config)
        results['v_projection'] = FoundationalOperators.linear_gemm_bias(v_config)
        
        # RoPE
        results['rope_q'] = DerivedOperators.rope(seq_len, num_q_heads, head_dim, dtype)
        results['rope_k'] = DerivedOperators.rope(seq_len, num_kv_heads, head_dim, dtype)
        
        # K,V repetition for GQA (broadcasting/indexing operations)
        kv_repeat_ops = seq_len * num_q_heads * head_dim * 2  # for K and V
        results['kv_repetition'] = (kv_repeat_ops, 0, 0)  # No additional memory I/O
        
        # Attention computation (same as MHA after repetition)
        qk_config = OperatorConfig((num_q_heads, seq_len, head_dim), (num_q_heads, head_dim, seq_len), dtype, dtype)
        results['attention_scores'] = FoundationalOperators.bmm(qk_config)
        
        results['softmax'] = DerivedOperators.softmax(seq_len, num_q_heads, dtype)
        
        av_config = OperatorConfig((num_q_heads, seq_len, seq_len), (num_q_heads, seq_len, head_dim), dtype, dtype)
        results['attention_output'] = FoundationalOperators.bmm(av_config)
        
        # Output projection
        o_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, hidden_dim), dtype, dtype)
        results['output_projection'] = FoundationalOperators.linear_gemm_bias(o_config)
        
        return results
    
    @staticmethod
    def swiglu_mlp(seq_len: int, hidden_dim: int, intermediate_dim: int, dtype: DataType) -> Dict[str, Tuple[int, int, int]]:
        """
        SwiGLU MLP as used in Llama: gate_proj(x) * silu(up_proj(x)) -> down_proj
        """
        results = {}
        
        # Gate projection
        gate_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, intermediate_dim), dtype, dtype)
        results['gate_projection'] = FoundationalOperators.linear_gemm_bias(gate_config)
        
        # Up projection  
        up_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, intermediate_dim), dtype, dtype)
        results['up_projection'] = FoundationalOperators.linear_gemm_bias(up_config)
        
        # SiLU activation on gate
        silu_config = OperatorConfig((seq_len, intermediate_dim), dtype_a=dtype)
        results['silu_activation'] = FoundationalOperators.elementwise(silu_config)
        
        # Element-wise multiplication: silu(gate) * up
        mult_config = OperatorConfig((seq_len, intermediate_dim), dtype_a=dtype)
        results['elementwise_multiply'] = FoundationalOperators.elementwise(mult_config)
        
        # Down projection
        down_config = OperatorConfig((seq_len, intermediate_dim), (intermediate_dim, hidden_dim), dtype, dtype)
        results['down_projection'] = FoundationalOperators.linear_gemm_bias(down_config)
        
        return results


class LLMWorkloadAnalyzer:
    """Analyze complete LLM workloads using operator-level analysis."""
    
    def __init__(self):
        self.operator_stats = {}
    
    def analyze_transformer_layer(self, config: Dict) -> Dict[str, Dict]:
        """
        Analyze a single transformer layer.
        """
        seq_len = config['sequence_length']
        hidden_dim = config['hidden_size']
        intermediate_dim = config['intermediate_size']
        num_q_heads = config['num_attention_heads']
        num_kv_heads = config.get('num_key_value_heads', num_q_heads)
        dtype = DataType[config.get('dtype', 'FP16')]
        
        layer_stats = {}
        
        # Pre-attention RMSNorm
        layer_stats['pre_attn_norm'] = DerivedOperators.rms_norm(seq_len, hidden_dim, dtype)
        
        # Attention
        if num_kv_heads == num_q_heads:
            # Standard MHA
            layer_stats['attention'] = DerivedOperators.mha_attention(seq_len, hidden_dim, num_q_heads, dtype)
        else:
            # GQA
            layer_stats['attention'] = DerivedOperators.gqa_attention(seq_len, hidden_dim, num_q_heads, num_kv_heads, dtype)
        
        # Post-attention RMSNorm
        layer_stats['post_attn_norm'] = DerivedOperators.rms_norm(seq_len, hidden_dim, dtype)
        
        # MLP (SwiGLU)
        layer_stats['mlp'] = DerivedOperators.swiglu_mlp(seq_len, hidden_dim, intermediate_dim, dtype)
        
        # Residual connections (2 per layer)
        residual_config = OperatorConfig((seq_len, hidden_dim), dtype_a=dtype)
        layer_stats['residual_1'] = FoundationalOperators.elementwise(residual_config)
        layer_stats['residual_2'] = FoundationalOperators.elementwise(residual_config)
        
        return layer_stats
    
    def analyze_full_model(self, config: Dict) -> Dict:
        """
        Analyze complete model including embedding and all layers.
        """
        seq_len = config['sequence_length']
        hidden_dim = config['hidden_size']
        vocab_size = config['vocab_size']
        num_layers = config['num_hidden_layers']
        dtype = DataType[config.get('dtype', 'FP16')]
        
        model_stats = {}
        
        # Token embedding
        model_stats['token_embedding'] = FoundationalOperators.embedding(vocab_size, hidden_dim, dtype)
        
        # All transformer layers
        layer_stats = self.analyze_transformer_layer(config)
        model_stats['layers'] = {
            'single_layer': layer_stats,
            'num_layers': num_layers,
            'total_layer_stats': self._multiply_stats(layer_stats, num_layers)
        }
        
        # Final RMSNorm
        model_stats['final_norm'] = DerivedOperators.rms_norm(seq_len, hidden_dim, dtype)
        
        # Output projection (LM head)
        output_config = OperatorConfig((seq_len, hidden_dim), (hidden_dim, vocab_size), dtype, dtype)
        model_stats['lm_head'] = FoundationalOperators.linear_gemm_bias(output_config)
        
        return model_stats
    
    def _multiply_stats(self, stats: Dict, multiplier: int) -> Dict:
        """Multiply all statistics by a given factor."""
        result = {}
        for key, value in stats.items():
            if isinstance(value, dict):
                result[key] = self._multiply_stats(value, multiplier)
            elif isinstance(value, tuple) and len(value) == 3:
                # (compute_ops, mem_read, mem_write)
                result[key] = (value[0] * multiplier, value[1] * multiplier, value[2] * multiplier)
            else:
                result[key] = value
        return result
    
    def summarize_stats(self, stats: Dict, name: str = "Model") -> None:
        """Print a summary of statistics."""
        total_compute = 0
        total_mem_read = 0
        total_mem_write = 0
        
        def accumulate_stats(data, prefix=""):
            nonlocal total_compute, total_mem_read, total_mem_write
            
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"\n{prefix}{key.upper()}:")
                    accumulate_stats(value, prefix + "  ")
                elif isinstance(value, tuple) and len(value) == 3:
                    ops, mem_r, mem_w = value
                    total_compute += ops
                    total_mem_read += mem_r
                    total_mem_write += mem_w
                    print(f"{prefix}{key}: {ops:,} ops, {mem_r/1e6:.1f}MB read, {mem_w/1e6:.1f}MB write")
                elif key not in ['num_layers']:
                    print(f"{prefix}{key}: {value}")
        
        print(f"\n{'='*60}")
        print(f"{name} Statistics Summary")
        print(f"{'='*60}")
        
        accumulate_stats(stats)
        
        print(f"\n{'='*60}")
        print(f"TOTAL SUMMARY:")
        print(f"  Compute Operations: {total_compute:,}")
        print(f"  Memory Read: {total_mem_read/1e9:.2f} GB")
        print(f"  Memory Write: {total_mem_write/1e9:.2f} GB")
        print(f"  Total Memory I/O: {(total_mem_read + total_mem_write)/1e9:.2f} GB")
        print(f"{'='*60}")


# Example usage
if __name__ == "__main__":
    # Llama 3.2 1B configuration
    llama_config = {
        'sequence_length': 2048,
        'hidden_size': 2048,
        'intermediate_size': 8192,
        'num_attention_heads': 32,
        'num_key_value_heads': 8,  # GQA
        'num_hidden_layers': 16,
        'vocab_size': 128256,
        'dtype': 'FP16'
    }
    
    analyzer = LLMWorkloadAnalyzer()
    
    # Analyze single layer
    print("Analyzing single transformer layer...")
    layer_stats = analyzer.analyze_transformer_layer(llama_config)
    analyzer.summarize_stats(layer_stats, "Single Transformer Layer")
    
    # Analyze full model
    print("\n" + "="*80)
    print("Analyzing full model...")
    model_stats = analyzer.analyze_full_model(llama_config)
    analyzer.summarize_stats(model_stats, "Complete Llama 3.2 1B Model")