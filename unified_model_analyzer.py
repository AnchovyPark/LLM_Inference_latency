"""
Unified Model Analyzer
Combines model code parsing with operation calculation to provide complete analysis
Input: model Python files + config.json + sequence parameters
Output: Complete operation count analysis for LLM inference
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Import our existing components
from model_code_parser import ModelCodeParser
from life_inspired_calculator import LLMWorkloadAnalyzer, DataType, FoundationalOperators, DerivedOperators, OperatorConfig


@dataclass
class InferenceParameters:
    """Parameters for inference analysis."""
    sequence_length: int = 1024
    batch_size: int = 1
    phase: str = "PREFILL"  # PREFILL or DECODE
    context_length: int = 1024  # For DECODE phase
    dtype: DataType = DataType.FP16


@dataclass
class OperationResult:
    """Result of operation analysis."""
    operator_name: str
    operator_type: str  # foundational or derived
    usage_count: int
    compute_ops: int
    memory_read: int
    memory_write: int
    total_memory: int


class UnifiedModelAnalyzer:
    """Unified analyzer combining model parsing and operation calculation."""
    
    def __init__(self):
        self.parser = ModelCodeParser()
        self.calculator = LLMWorkloadAnalyzer()
    
    def analyze_model(
        self, 
        model_path: str, 
        config_path: str, 
        inference_params: InferenceParameters
    ) -> Dict[str, Any]:
        """
        Complete model analysis combining code parsing and operation calculation.
        
        Args:
            model_path: Path to model Python file(s)
            config_path: Path to config.json
            inference_params: Inference parameters (sequence length, batch size, etc.)
        
        Returns:
            Complete analysis results
        """
        
        print(f"🎯 UNIFIED MODEL ANALYSIS")
        print(f"Model: {model_path}")
        print(f"Config: {config_path}")
        print(f"Sequence: {inference_params.sequence_length}, Batch: {inference_params.batch_size}")
        print(f"Phase: {inference_params.phase}")
        print("=" * 70)
        
        # Step 1: Parse model architecture and code
        arch_params = self.parser.parse_config_file(config_path)
        operator_usage = self._parse_model_files(model_path)
        
        # Step 2: Calculate theoretical operator counts based on architecture
        theoretical_counts = self.parser.calculate_theoretical_operator_counts(
            arch_params, inference_params.sequence_length
        )
        
        # Step 3: Calculate actual operations for each operator
        operation_results = self._calculate_operations(
            arch_params, 
            theoretical_counts, 
            inference_params
        )
        
        # Step 4: Generate comprehensive analysis
        analysis = {
            'model_architecture': arch_params,
            'inference_parameters': {
                'sequence_length': inference_params.sequence_length,
                'batch_size': inference_params.batch_size,
                'phase': inference_params.phase,
                'context_length': inference_params.context_length,
                'dtype': inference_params.dtype.name
            },
            'parsed_operators': operator_usage,
            'theoretical_counts': theoretical_counts,
            'operation_results': operation_results,
            'totals': self._calculate_totals(operation_results)
        }
        
        return analysis
    
    def _parse_model_files(self, model_path: str) -> Dict[str, Any]:
        """Parse model implementation files."""
        
        model_dir = Path(model_path)
        if model_dir.is_file():
            # Single file
            return self.parser.parse_python_file(str(model_dir))
        else:
            # Directory - parse all Python files
            python_files = list(model_dir.glob("**/*.py"))
            print(f"Found {len(python_files)} Python files to parse")
            
            all_usage = {}
            for py_file in python_files:
                file_usage = self.parser.parse_python_file(str(py_file))
                # Merge usage counts
                for key, usage in file_usage.items():
                    if key in all_usage:
                        all_usage[key].count += usage.count
                        all_usage[key].locations.extend(usage.locations)
                    else:
                        all_usage[key] = usage
            
            return all_usage
    
    def _calculate_operations(
        self, 
        arch_params: Dict[str, Any], 
        theoretical_counts: Dict[str, int],
        inference_params: InferenceParameters
    ) -> List[OperationResult]:
        """Calculate actual operations for each operator type."""
        
        results = []
        
        # Extract architecture parameters
        layers = arch_params.get('num_hidden_layers', 16)
        hidden_size = arch_params.get('hidden_size', 2048)
        intermediate_size = arch_params.get('intermediate_size', 5632)
        num_q_heads = arch_params.get('num_attention_heads', 32)
        num_kv_heads = arch_params.get('num_key_value_heads', 8)
        vocab_size = arch_params.get('vocab_size', 128256)
        head_dim = hidden_size // num_q_heads
        
        seq_len = inference_params.sequence_length
        batch_size = inference_params.batch_size
        dtype = inference_params.dtype
        
        # Calculate operations for each operator type
        for op_key, count in theoretical_counts.items():
            if 'foundational_Linear' in op_key:
                # Linear operations: Q, K, V, O projections + MLP projections
                # Attention projections (4 per layer)
                attn_linear_ops = 0
                attn_linear_memory = 0
                
                # Q projection: [batch, seq, hidden] -> [batch, seq, num_q_heads * head_dim]
                q_config = OperatorConfig(
                    shape_a=(batch_size * seq_len, hidden_size),
                    shape_b=(hidden_size, num_q_heads * head_dim),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                q_ops, q_read, q_write = FoundationalOperators.linear_gemm_bias(q_config)
                
                # K projection: [batch, seq, hidden] -> [batch, seq, num_kv_heads * head_dim]
                k_config = OperatorConfig(
                    shape_a=(batch_size * seq_len, hidden_size),
                    shape_b=(hidden_size, num_kv_heads * head_dim),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                k_ops, k_read, k_write = FoundationalOperators.linear_gemm_bias(k_config)
                
                # V projection: [batch, seq, hidden] -> [batch, seq, num_kv_heads * head_dim]
                v_config = OperatorConfig(
                    shape_a=(batch_size * seq_len, hidden_size),
                    shape_b=(hidden_size, num_kv_heads * head_dim),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                v_ops, v_read, v_write = FoundationalOperators.linear_gemm_bias(v_config)
                
                # O projection: [batch, seq, num_q_heads * head_dim] -> [batch, seq, hidden]
                o_config = OperatorConfig(
                    shape_a=(batch_size * seq_len, num_q_heads * head_dim),
                    shape_b=(num_q_heads * head_dim, hidden_size),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                o_ops, o_read, o_write = FoundationalOperators.linear_gemm_bias(o_config)
                
                attn_linear_ops = (q_ops + k_ops + v_ops + o_ops) * layers
                attn_linear_memory = (q_read + q_write + k_read + k_write + 
                                    v_read + v_write + o_read + o_write) * layers
                
                # MLP projections (3 per layer: gate, up, down)
                mlp_linear_ops = 0
                mlp_linear_memory = 0
                
                # Gate & Up projections: [batch, seq, hidden] -> [batch, seq, intermediate]
                gate_config = OperatorConfig(
                    shape_a=(batch_size * seq_len, hidden_size),
                    shape_b=(hidden_size, intermediate_size),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                gate_ops, gate_read, gate_write = FoundationalOperators.linear_gemm_bias(gate_config)
                
                up_config = OperatorConfig(
                    shape_a=(batch_size * seq_len, hidden_size),
                    shape_b=(hidden_size, intermediate_size),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                up_ops, up_read, up_write = FoundationalOperators.linear_gemm_bias(up_config)
                
                # Down projection: [batch, seq, intermediate] -> [batch, seq, hidden]
                down_config = OperatorConfig(
                    shape_a=(batch_size * seq_len, intermediate_size),
                    shape_b=(intermediate_size, hidden_size),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                down_ops, down_read, down_write = FoundationalOperators.linear_gemm_bias(down_config)
                
                mlp_linear_ops = (gate_ops + up_ops + down_ops) * layers
                mlp_linear_memory = (gate_read + gate_write + up_read + up_write + 
                                   down_read + down_write) * layers
                
                total_ops = attn_linear_ops + mlp_linear_ops
                total_memory = attn_linear_memory + mlp_linear_memory
                
                results.append(OperationResult(
                    operator_name="Linear",
                    operator_type="foundational",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_memory // 2,  # Approximate split
                    memory_write=total_memory // 2,
                    total_memory=total_memory
                ))
                
            elif 'foundational_BMM' in op_key:
                # Batch matrix multiplication in attention
                # Q @ K^T: [batch * heads, seq, head_dim] @ [batch * heads, head_dim, seq]
                qk_config = OperatorConfig(
                    shape_a=(batch_size * num_q_heads, seq_len, head_dim),
                    shape_b=(batch_size * num_q_heads, head_dim, seq_len),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                qk_ops, qk_read, qk_write = FoundationalOperators.bmm(qk_config)
                
                # Attn @ V: [batch * heads, seq, seq] @ [batch * heads, seq, head_dim]
                av_config = OperatorConfig(
                    shape_a=(batch_size * num_q_heads, seq_len, seq_len),
                    shape_b=(batch_size * num_q_heads, seq_len, head_dim),
                    dtype_a=dtype, dtype_b=dtype, dtype_out=dtype
                )
                av_ops, av_read, av_write = FoundationalOperators.bmm(av_config)
                
                total_ops = (qk_ops + av_ops) * layers
                total_memory = (qk_read + qk_write + av_read + av_write) * layers
                
                results.append(OperationResult(
                    operator_name="BMM",
                    operator_type="foundational",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_memory // 2,
                    memory_write=total_memory // 2,
                    total_memory=total_memory
                ))
                
            elif 'foundational_Elementwise' in op_key:
                # Elementwise operations: adds, muls, activations
                # Estimate based on model operations
                element_count = batch_size * seq_len * hidden_size
                ops_per_layer = element_count * 8  # Multiple elementwise ops per layer
                
                total_ops = ops_per_layer * layers
                total_memory = element_count * dtype.value * 4 * layers
                
                results.append(OperationResult(
                    operator_name="Elementwise",
                    operator_type="foundational",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_memory // 2,
                    memory_write=total_memory // 2,
                    total_memory=total_memory
                ))
                
            elif 'foundational_Embedding' in op_key:
                # Token embedding lookup - use direct method call
                emb_ops, emb_read, emb_write = FoundationalOperators.embedding(
                    vocab_size, hidden_size, dtype
                )
                # Scale by batch size and sequence length
                emb_ops *= batch_size * seq_len
                emb_read *= batch_size * seq_len  
                emb_write *= batch_size * seq_len
                
                results.append(OperationResult(
                    operator_name="Embedding",
                    operator_type="foundational",
                    usage_count=count,
                    compute_ops=emb_ops,
                    memory_read=emb_read,
                    memory_write=emb_write,
                    total_memory=emb_read + emb_write
                ))
                
            elif 'derived_GQA' in op_key or 'derived_MHA' in op_key:
                # Grouped Query Attention or Multi-Head Attention
                attention_type = 'GQA' if 'GQA' in op_key else 'MHA'
                
                if attention_type == 'GQA':
                    # Use actual GQA calculation from DerivedOperators
                    gqa_results = DerivedOperators.gqa_attention(
                        seq_len, hidden_size, num_q_heads, num_kv_heads, dtype
                    )
                    # gqa_results is a Dict with operation breakdown
                    total_gqa_ops = sum(ops for ops, _, _ in gqa_results.values())
                    total_gqa_read = sum(read for _, read, _ in gqa_results.values())
                    total_gqa_write = sum(write for _, _, write in gqa_results.values())
                else:
                    # Use actual MHA calculation from DerivedOperators
                    mha_results = DerivedOperators.mha_attention(
                        seq_len, hidden_size, num_q_heads, dtype
                    )
                    # mha_results is a Dict with operation breakdown
                    total_gqa_ops = sum(ops for ops, _, _ in mha_results.values())
                    total_gqa_read = sum(read for _, read, _ in mha_results.values())
                    total_gqa_write = sum(write for _, _, write in mha_results.values())
                
                # Apply batch size scaling
                total_ops = total_gqa_ops * batch_size * layers
                total_read = total_gqa_read * batch_size * layers  
                total_write = total_gqa_write * batch_size * layers
                
                results.append(OperationResult(
                    operator_name=attention_type,
                    operator_type="derived",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_read,
                    memory_write=total_write,
                    total_memory=total_read + total_write
                ))
                
            elif 'derived_MLP' in op_key:
                # SwiGLU MLP block using exact calculation
                mlp_results = DerivedOperators.swiglu_mlp(
                    seq_len, hidden_size, intermediate_size, dtype
                )
                # mlp_results is a Dict with operation breakdown
                total_mlp_ops = sum(ops for ops, _, _ in mlp_results.values())
                total_mlp_read = sum(read for _, read, _ in mlp_results.values())
                total_mlp_write = sum(write for _, _, write in mlp_results.values())
                
                # Apply batch size and layer scaling
                total_ops = total_mlp_ops * batch_size * layers
                total_read = total_mlp_read * batch_size * layers
                total_write = total_mlp_write * batch_size * layers
                
                results.append(OperationResult(
                    operator_name="MLP",
                    operator_type="derived",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_read,
                    memory_write=total_write,
                    total_memory=total_read + total_write
                ))
                
            elif 'derived_RMSNorm' in op_key:
                # RMS Normalization using exact calculation
                norm_ops, norm_read, norm_write = DerivedOperators.rms_norm(
                    seq_len, hidden_size, dtype
                )
                
                # Apply batch size and actual usage count
                total_ops = norm_ops * batch_size * count
                total_read = norm_read * batch_size * count  
                total_write = norm_write * batch_size * count
                
                results.append(OperationResult(
                    operator_name="RMSNorm",
                    operator_type="derived",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_read,
                    memory_write=total_write,
                    total_memory=total_read + total_write
                ))
                
            elif 'derived_Softmax' in op_key:
                # Softmax in attention - simplified calculation
                # Softmax over [batch, heads, seq, seq] 
                softmax_elements = batch_size * num_q_heads * seq_len * seq_len
                softmax_ops = softmax_elements * 5  # exp + sum + div operations
                softmax_memory = softmax_elements * dtype.value * 2  # input + output
                
                total_ops = softmax_ops * layers
                total_read = softmax_memory * layers // 2
                total_write = softmax_memory * layers // 2
                
                results.append(OperationResult(
                    operator_name="Softmax",
                    operator_type="derived",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_read,
                    memory_write=total_write,
                    total_memory=total_read + total_write
                ))
                
            # Add RoPE if present (simple elementwise calculation)
            elif 'derived_RoPE' in op_key:
                # Rotary Position Embedding - elementwise operations
                rope_elements = batch_size * seq_len * num_q_heads * head_dim
                rope_ops = rope_elements * 4  # cos, sin, multiply operations
                rope_memory = rope_elements * dtype.value * 2
                
                total_ops = rope_ops * layers
                total_read = rope_memory * layers // 2
                total_write = rope_memory * layers // 2
                
                results.append(OperationResult(
                    operator_name="RoPE", 
                    operator_type="derived",
                    usage_count=count,
                    compute_ops=total_ops,
                    memory_read=total_read,
                    memory_write=total_write,
                    total_memory=total_read + total_write
                ))
        
        return results
    
    def _calculate_totals(self, operation_results: List[OperationResult]) -> Dict[str, Any]:
        """Calculate total statistics across all operations."""
        
        total_compute_ops = sum(r.compute_ops for r in operation_results)
        total_memory = sum(r.total_memory for r in operation_results)
        
        foundational_ops = [r for r in operation_results if r.operator_type == "foundational"]
        derived_ops = [r for r in operation_results if r.operator_type == "derived"]
        
        return {
            'total_compute_ops': total_compute_ops,
            'total_memory_bytes': total_memory,
            'foundational_compute_ops': sum(r.compute_ops for r in foundational_ops),
            'derived_compute_ops': sum(r.compute_ops for r in derived_ops),
            'foundational_memory': sum(r.total_memory for r in foundational_ops),
            'derived_memory': sum(r.total_memory for r in derived_ops),
            'operation_breakdown': {
                'foundational_count': len(foundational_ops),
                'derived_count': len(derived_ops),
                'total_count': len(operation_results)
            }
        }
    
    def print_analysis_summary(self, analysis: Dict[str, Any]):
        """Print comprehensive analysis summary."""
        
        print(f"\n🎯 UNIFIED ANALYSIS RESULTS")
        print("=" * 70)
        
        # Model info
        arch = analysis['model_architecture']
        params = analysis['inference_parameters']
        
        print(f"\n📐 Model Configuration:")
        print(f"   Architecture: {arch.get('model_type', 'unknown')}")
        print(f"   Layers: {arch.get('num_hidden_layers', 0)}")
        print(f"   Hidden Size: {arch.get('hidden_size', 0):,}")
        print(f"   Attention: {arch.get('attention_type', 'unknown')} "
              f"({arch.get('num_attention_heads', 0)}Q:{arch.get('num_key_value_heads', 0)}KV)")
        print(f"   Vocab Size: {arch.get('vocab_size', 0):,}")
        
        print(f"\n⚙️  Inference Parameters:")
        print(f"   Sequence Length: {params['sequence_length']:,}")
        print(f"   Batch Size: {params['batch_size']}")
        print(f"   Phase: {params['phase']}")
        print(f"   Data Type: {params['dtype']}")
        
        # Operation results
        print(f"\n🔧 OPERATION ANALYSIS:")
        print(f"{'Operator':<15} {'Type':<12} {'Count':<8} {'Compute Ops':<15} {'Memory (MB)':<12}")
        print("-" * 75)
        
        for result in analysis['operation_results']:
            memory_mb = result.total_memory / (1024 * 1024)
            compute_gops = result.compute_ops / 1e9
            
            print(f"{result.operator_name:<15} {result.operator_type:<12} "
                  f"{result.usage_count:<8} {compute_gops:<15.2f} {memory_mb:<12.2f}")
        
        # Totals
        totals = analysis['totals']
        print(f"\n📊 TOTALS:")
        print(f"   Total Compute Ops: {totals['total_compute_ops']/1e12:.2f} TOPS")
        print(f"   Total Memory: {totals['total_memory_bytes']/(1024**3):.2f} GB")
        print(f"   Foundational Ops: {totals['foundational_compute_ops']/1e12:.2f} TOPS")
        print(f"   Derived Ops: {totals['derived_compute_ops']/1e12:.2f} TOPS")
        
        # Arithmetic intensity
        ai = totals['total_compute_ops'] / totals['total_memory_bytes']
        print(f"   Arithmetic Intensity: {ai:.2f} ops/byte")


def main():
    """Demonstration of unified model analysis."""
    
    analyzer = UnifiedModelAnalyzer()
    
    # Configure analysis parameters
    model_path = "/Users/anchovy-mac/Desktop/calculating/mock_llama_model.py"
    config_path = "/Users/anchovy-mac/Desktop/calculating/config.json"
    
    inference_params = InferenceParameters(
        sequence_length=1024,
        batch_size=1,
        phase="PREFILL",
        dtype=DataType.FP16
    )
    
    # Run complete analysis
    analysis = analyzer.analyze_model(model_path, config_path, inference_params)
    
    # Print results
    analyzer.print_analysis_summary(analysis)
    
    print(f"\n💡 Analysis Complete!")
    print(f"   ✓ Parsed model implementation")
    print(f"   ✓ Calculated theoretical operator counts")
    print(f"   ✓ Computed actual operation counts")
    print(f"   ✓ Generated complete performance profile")


if __name__ == "__main__":
    main()