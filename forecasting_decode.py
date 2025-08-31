"""
Forecasting Decode Calculator

This module contains latency calculation formulas for LLM decoding phase,
focusing on operations that occur during token generation.
"""

from dataclasses import dataclass
from typing import Optional
from delay_factor_manager import DelayFactorManager


class ModelConfig:
    """LLM model configuration with predefined model specs."""
    
    def __init__(self, name: str, hidden_size: int, intermediate_size: int, 
                 num_attention_heads: int, num_key_value_heads: int, 
                 head_dim: int, num_layers: int, vocab_size: int):
        self.name = name
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
    
    @staticmethod
    def llama_3_2_1B():
        """LLaMA 3.2 1B model configuration."""
        return ModelConfig(
            name="LLaMA-3.2-1B",
            hidden_size=2048,
            intermediate_size=8192,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=64,
            num_layers=16,
            vocab_size=128256
        )
    
    @staticmethod
    def llama_3_2_3B():
        """LLaMA 3.2 3B model configuration."""
        return ModelConfig(
            name="LLaMA-3.2-3B",
            hidden_size=3072,
            intermediate_size=8192,
            num_attention_heads=24,
            num_key_value_heads=8,
            head_dim=128,
            num_layers=28,
            vocab_size=128256
        )
    
    @staticmethod
    def llama_3_0_8B():
        """LLaMA 3.0 8B model configuration."""
        return ModelConfig(
            name="LLaMA-3.0-8B",
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_layers=32,
            vocab_size=128256
        )
    
    @staticmethod
    def llama_3_1_8B():
        """LLaMA 3.1 8B model configuration."""
        return ModelConfig(
            name="LLaMA-3.1-8B",
            hidden_size=4096,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_layers=32,
            vocab_size=128256
        )
    
    @staticmethod
    def llama_3_0_70B():
        """LLaMA 3.0 70B model configuration."""
        return ModelConfig(
            name="LLaMA-3.0-70B",
            hidden_size=8192,
            intermediate_size=28672,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            num_layers=80,
            vocab_size=128256
        )
    
    @staticmethod
    def llama_3_1_70B():
        """LLaMA 3.1 70B model configuration."""
        return ModelConfig(
            name="LLaMA-3.1-70B",
            hidden_size=8192,
            intermediate_size=28672,
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            num_layers=80,
            vocab_size=128256
        )
    
    @staticmethod
    def llama_3_1_405B():
        """LLaMA 3.1 405B model configuration."""
        return ModelConfig(
            name="LLaMA-3.1-405B",
            hidden_size=16384,
            intermediate_size=53248,
            num_attention_heads=128,
            num_key_value_heads=16,
            head_dim=128,
            num_layers=126,
            vocab_size=128256
        )


class GPUConfig:
    """GPU configuration with predefined GPU specs."""
    
    def __init__(self, name: str, peak_fp16_tflops: float, memory_bandwidth_gb_s: float):
        self.name = name
        self.peak_fp16_tflops = peak_fp16_tflops
        self.memory_bandwidth_gb_s = memory_bandwidth_gb_s
    
    @staticmethod
    def T4():
        """NVIDIA Tesla T4 GPU configuration."""
        return GPUConfig(
            name="T4",
            peak_fp16_tflops=65.13,
            memory_bandwidth_gb_s=300.0
        )
    
    @staticmethod
    def A10G():
        """NVIDIA A10G GPU configuration."""
        return GPUConfig(
            name="A10G", 
            peak_fp16_tflops=125,
            memory_bandwidth_gb_s=600.0
        )
    
    @staticmethod
    def L4():
        """NVIDIA L4 GPU configuration."""
        return GPUConfig(
            name="L4",
            peak_fp16_tflops=121.0,
            memory_bandwidth_gb_s=300.0
        )
    
    @staticmethod
    def L40():
        """NVIDIA L40 GPU configuration."""
        return GPUConfig(
            name="L40",
            peak_fp16_tflops=181.05,
            memory_bandwidth_gb_s=864.0
        )
    
    @staticmethod
    def L40S():
        """NVIDIA L40S GPU configuration."""
        return GPUConfig(
            name="L40S",
            peak_fp16_tflops=362.07,
            memory_bandwidth_gb_s=864.0
        )
    
    @staticmethod
    def A100_40GB():
        """NVIDIA A100 40GB GPU configuration."""
        return GPUConfig(
            name="A100-40GB",
            peak_fp16_tflops=77.97,
            memory_bandwidth_gb_s=1555.0
        )
    
    @staticmethod  
    def A100_80GB():
        """NVIDIA A100 80GB GPU configuration."""
        return GPUConfig(
            name="A100-80GB",
            peak_fp16_tflops=77.97,
            memory_bandwidth_gb_s=1935.0
        )
    
    @staticmethod
    def H100_PCIe():
        """NVIDIA H100 PCIe GPU configuration."""
        return GPUConfig(
            name="H100-PCIe",
            peak_fp16_tflops=204.9,
            memory_bandwidth_gb_s=2000.0
        )
    
    @staticmethod
    def H100_SXM():
        """NVIDIA H100 SXM GPU configuration."""
        return GPUConfig(
            name="H100-SXM",
            peak_fp16_tflops=267.6,
            memory_bandwidth_gb_s=3350.0
        )


@dataclass
class DecodingLatencyComponents:
    """Stores all calculated decoding latency components."""
    embed_flops: float
    embed_r_bytes: float
    embed_w_bytes: float
    embed_bytes: float

    q_proj_flops: float
    q_proj_r_bytes: float
    q_proj_w_bytes: float
    q_proj_bytes: float

    kv_proj_flops: float
    kv_proj_r_bytes: float
    kv_proj_w_bytes: float
    kv_proj_bytes: float

    rope_flops: float
    rope_r_bytes: float
    rope_w_bytes: float
    rope_bytes: float

    mlp_flops: float
    mlp_r_bytes: float
    mlp_w_bytes: float
    mlp_bytes: float

    attention_qk_flops: float
    attention_qk_r_bytes: float
    attention_qk_w_bytes: float
    attention_qk_bytes: float

    attention_v_flops: float
    attention_v_r_bytes: float
    attention_v_w_bytes: float
    attention_v_bytes: float

    attention_o_flops: float
    attention_o_r_bytes: float
    attention_o_w_bytes: float
    attention_o_bytes: float

    mask_flops: float
    mask_r_bytes: float
    mask_w_bytes: float
    mask_bytes: float

    softmax_flops: float
    softmax_r_bytes: float
    softmax_w_bytes: float
    softmax_bytes: float

    residual_flops: float
    residual_r_bytes: float
    residual_w_bytes: float
    residual_bytes: float

    norm_flops: float
    norm_r_bytes: float
    norm_w_bytes: float
    norm_bytes: float

    lm_head_flops: float
    lm_head_r_bytes: float
    lm_head_w_bytes: float
    lm_head_bytes: float

    generate_flops: float
    generate_r_bytes: float
    generate_w_bytes: float
    generate_bytes: float

    kv_cache_store_flops: float
    kv_cache_store_r_bytes: float
    kv_cache_store_w_bytes: float
    kv_cache_store_bytes: float


class ForecastingDecodeCalculator:
    """Calculate latency components for LLM decoding phase."""
    
    def __init__(self, model_config: ModelConfig, gpu_config: GPUConfig, 
                 delay_factor_config_file: Optional[str] = None):
        """
        Initialize the calculator with model and GPU configurations.
        
        Args:
            model_config: Model configuration object
            gpu_config: GPU configuration object
            delay_factor_config_file: Path to delay factor configuration JSON file
        """
        self.model = model_config
        self.gpu = gpu_config
        self.bytes_per_element = 2  # Assuming FP16
        
        # Convert GPU specs to proper units
        self.peak_tflops = gpu_config.peak_fp16_tflops * 1e12
        self.memory_bw_bytes = gpu_config.memory_bandwidth_gb_s * 1e9
        
        # Initialize delay factor manager
        self.delay_factor_manager = DelayFactorManager(
            gpu_name=gpu_config.name,
            config_file=delay_factor_config_file
        )

    # =====================================================================
    # FIXED LATENCY OPERATIONS (토큰 길이와 무관한 고정 연산들)
    # =====================================================================
    
    def calculate_embed_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate embedding operations for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        embed_flops = 0
        embed_r_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element
        embed_w_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element
        
        return embed_flops, embed_r_bytes, embed_w_bytes
    
    def calculate_q_projection_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Q projection FLOPs and bytes for decoding (single token generation).
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        q_proj_flops = (model.hidden_size * model.hidden_size) * batch_size * seq_len * 2
        
        q_proj_r_bytes = (
            (model.hidden_size * model.hidden_size) +  # Q weight matrix read
            (batch_size * seq_len * model.hidden_size)  # Input read
        ) * self.bytes_per_element
        
        q_proj_w_bytes = (
            batch_size * model.num_attention_heads * seq_len * model.head_dim  # Q output write
        ) * self.bytes_per_element
        
        return q_proj_flops, q_proj_r_bytes, q_proj_w_bytes
    
    def calculate_kv_projection_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate KV projection FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        kv_proj_flops = (
            (model.hidden_size * model.head_dim * model.num_key_value_heads) * 2  # K, V
        ) * batch_size * seq_len * 2
        
        kv_proj_r_bytes = (
            (model.head_dim * model.num_key_value_heads * model.hidden_size) * 2 +  # K,V weight matrices read
            (batch_size * seq_len * model.hidden_size) * 2  # Input read (for K and V)
        ) * self.bytes_per_element
        
        kv_proj_w_bytes = (
            batch_size * model.num_key_value_heads * seq_len * model.head_dim * 2  # KV output write
        ) * self.bytes_per_element
        
        return kv_proj_flops, kv_proj_r_bytes, kv_proj_w_bytes
    
    def calculate_mlp_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate MLP (FFN) Projections FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        mlp_flops = (
            2 * (model.hidden_size * model.intermediate_size) * 2 +  # Gate, Up
            2 * (model.intermediate_size * model.hidden_size) +      # Down
            (model.intermediate_size) * 2  # element wise, Act_Fun
        ) * batch_size * seq_len
        
        mlp_r_bytes = (
            (batch_size * seq_len * model.hidden_size) +  # Read hidden_states
            (model.intermediate_size * model.hidden_size) * 3  # Read Gate, Up, Down weight matrices
        ) * self.bytes_per_element
        
        mlp_w_bytes = (
            batch_size * seq_len * model.hidden_size  # Write hidden_states
        ) * self.bytes_per_element
        
        return mlp_flops, mlp_r_bytes, mlp_w_bytes
    
    def calculate_attention_o_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Weights_O projection FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        attention_o_flops = (
            batch_size * seq_len * model.hidden_size**2 * 2
        )
        
        attention_o_r_bytes = (
            (batch_size * seq_len * model.hidden_size) +  # Read input
            (model.hidden_size**2)  # Read O weight matrix
        ) * self.bytes_per_element
        
        attention_o_w_bytes = (
            batch_size * seq_len * model.hidden_size  # Write output
        ) * self.bytes_per_element
        
        return attention_o_flops, attention_o_r_bytes, attention_o_w_bytes
    
    def calculate_residual_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Residual Connection FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        residual_flops = (
            batch_size * seq_len * model.hidden_size * 2
        )
        
        residual_r_bytes = (
            batch_size * seq_len * model.hidden_size * 2 * 2  # 2 inputs × 2 residual connections
        ) * self.bytes_per_element
        
        residual_w_bytes = (
            batch_size * seq_len * model.hidden_size * 2  # 2 outputs (one per residual connection)
        ) * self.bytes_per_element
        
        return residual_flops, residual_r_bytes, residual_w_bytes
    
    def calculate_norm_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate RMSNorm FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        norm_flops = batch_size * seq_len * model.hidden_size * 3
        
        norm_r_bytes = (
            (batch_size * seq_len * model.hidden_size) +  # Read input
            model.hidden_size  # Read norm weights
        ) * self.bytes_per_element
        
        norm_w_bytes = (
            batch_size * seq_len * model.hidden_size  # Write normalized output
        ) * self.bytes_per_element
        
        return norm_flops, norm_r_bytes, norm_w_bytes
    
    def calculate_lm_head_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate LM Head Projection FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        lm_head_flops = batch_size * seq_len * model.hidden_size * model.vocab_size * 2
        
        lm_head_r_bytes = (
            batch_size * seq_len * model.hidden_size +  # Read input
            model.hidden_size * model.vocab_size  # Read weight matrix
        ) * self.bytes_per_element
        
        lm_head_w_bytes = (
            batch_size * seq_len * model.vocab_size  # Write logits
        ) * self.bytes_per_element
        
        return lm_head_flops, lm_head_r_bytes, lm_head_w_bytes
    
    def calculate_generate_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate generate (logits to token ID) operations for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        generate_flops = 0
        generate_r_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element
        generate_w_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element
        
        return generate_flops, generate_r_bytes, generate_w_bytes
    
    def calculate_kv_cache_store_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate KV cache storage FLOPs and bytes for decoding.
        This represents storing the new K,V values into the cache.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        
        # KV cache store has minimal compute overhead
        kv_cache_store_flops = 0
        
        # Read: New K,V values to store
        kv_cache_store_r_bytes = 0
        
        # Write: Store K,V values in cache
        kv_cache_store_w_bytes = (
            batch_size * seq_len * model.num_key_value_heads * model.head_dim * 2  # K,V values
        ) * self.bytes_per_element
        
        return kv_cache_store_flops, kv_cache_store_r_bytes, kv_cache_store_w_bytes
        
    # =====================================================================
    # DYNAMIC LATENCY OPERATIONS (KV 캐시 길이에 비례하는 동적 연산들)
    # =====================================================================
    
    def calculate_rope_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Rotary Position Embedding FLOPs and bytes for decoding.

        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        seq_len = 1  # Decoding generates one token at a time
        rope_seq_len = kv_cache_len + 1 
        
        rope_flops = (
            model.num_attention_heads + model.num_key_value_heads
        ) * batch_size * seq_len * model.head_dim * 3
        
        rope_r_bytes = (
            ((model.num_attention_heads + model.num_key_value_heads) * batch_size * seq_len * model.head_dim) +  # Q, K read
            (2 * rope_seq_len * model.head_dim)  # cos, sin read
        ) * self.bytes_per_element
        
        rope_w_bytes = (
            (model.num_attention_heads + model.num_key_value_heads) * batch_size * seq_len * model.head_dim
        ) * self.bytes_per_element
        
        return rope_flops, rope_r_bytes, rope_w_bytes
    
    def calculate_attention_qk_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Attention Q @ K FLOPs and bytes for decoding.
        Note: In decoding, Q is 1 token, K is entire cache (kv_cache_len)
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        q_seq_len = 1  # Current token being generated
        k_seq_len = kv_cache_len + 1  # KV cache + current token
        
        attention_qk_flops = (
            2 * batch_size * model.num_attention_heads * 
            q_seq_len * k_seq_len * model.head_dim
        )
        
        attention_qk_r_bytes = (
            (model.num_attention_heads * q_seq_len * model.head_dim * batch_size) +  # Read Q
            (model.num_key_value_heads * k_seq_len * model.head_dim * batch_size)  # Read K from cache
        ) * self.bytes_per_element
        
        attention_qk_w_bytes = (
            model.num_attention_heads * q_seq_len * k_seq_len * batch_size  # Write attention scores
        ) * self.bytes_per_element
        
        return attention_qk_flops, attention_qk_r_bytes, attention_qk_w_bytes
    
    def calculate_attention_mask_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Attention Mask FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        q_seq_len = 1  # Current token being generated
        k_seq_len = kv_cache_len + 1  # KV cache + current token
        
        mask_flops = (
            batch_size * model.num_attention_heads * q_seq_len * k_seq_len 
        )
        
        mask_r_bytes = (
            (batch_size * model.num_attention_heads * q_seq_len * k_seq_len) +  # Read attention scores
            (batch_size * q_seq_len * k_seq_len)  # Read mask
        ) * self.bytes_per_element
        
        mask_w_bytes = (
            batch_size * model.num_attention_heads * q_seq_len * k_seq_len  # Write masked scores
        ) * self.bytes_per_element
        
        return mask_flops, mask_r_bytes, mask_w_bytes
    
    def calculate_softmax_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Softmax FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        q_seq_len = 1  # Current token being generated
        k_seq_len = kv_cache_len + 1  # KV cache + current token
        
        softmax_flops = (
            batch_size * model.num_attention_heads * q_seq_len * k_seq_len * 3
        )
        
        softmax_r_bytes = (
            batch_size * model.num_attention_heads * q_seq_len * k_seq_len
        ) * self.bytes_per_element
        
        softmax_w_bytes = (
            batch_size * model.num_attention_heads * q_seq_len * k_seq_len
        ) * self.bytes_per_element
        
        return softmax_flops, softmax_r_bytes, softmax_w_bytes
    
    def calculate_attention_v_decode(self, batch_size: int, kv_cache_len: int) -> tuple[float, float, float]:
        """
        Calculate Attention weights @ V FLOPs and bytes for decoding.
        
        Args:
            batch_size: Batch size
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        model = self.model
        q_seq_len = 1  # Current token being generated
        v_seq_len = kv_cache_len + 1  # KV cache + current token
        
        attention_v_flops = (
            2 * batch_size * model.num_attention_heads * 
            q_seq_len * v_seq_len * model.head_dim
        )
        
        attention_v_r_bytes = (
            (q_seq_len * v_seq_len * model.num_attention_heads) +  # Read attention weights
            (v_seq_len * model.head_dim * model.num_key_value_heads)  # Read V from cache
        ) * batch_size * self.bytes_per_element
        
        attention_v_w_bytes = (
            q_seq_len * model.head_dim * model.num_attention_heads  # Write attention output
        ) * batch_size * self.bytes_per_element
        
        return attention_v_flops, attention_v_r_bytes, attention_v_w_bytes
        
    def calculate_all_decode_components(self, batch_size: int, kv_cache_len: int) -> DecodingLatencyComponents:
        """
        Calculate all decoding latency components at once.
        
        Args:
            batch_size: Batch size for inference
            kv_cache_len: Length of KV cache (past sequence length)
            
        Returns:
            DecodingLatencyComponents object with all calculated values
        """
        embed_flops, embed_r_bytes, embed_w_bytes = self.calculate_embed_decode(batch_size, kv_cache_len)
        q_proj_flops, q_proj_r_bytes, q_proj_w_bytes = self.calculate_q_projection_decode(batch_size, kv_cache_len)
        kv_proj_flops, kv_proj_r_bytes, kv_proj_w_bytes = self.calculate_kv_projection_decode(batch_size, kv_cache_len)
        rope_flops, rope_r_bytes, rope_w_bytes = self.calculate_rope_decode(batch_size, kv_cache_len)
        mlp_flops, mlp_r_bytes, mlp_w_bytes = self.calculate_mlp_decode(batch_size, kv_cache_len)
        attention_qk_flops, attention_qk_r_bytes, attention_qk_w_bytes = self.calculate_attention_qk_decode(batch_size, kv_cache_len)
        attention_v_flops, attention_v_r_bytes, attention_v_w_bytes = self.calculate_attention_v_decode(batch_size, kv_cache_len)
        attention_o_flops, attention_o_r_bytes, attention_o_w_bytes = self.calculate_attention_o_decode(batch_size, kv_cache_len)
        mask_flops, mask_r_bytes, mask_w_bytes = self.calculate_attention_mask_decode(batch_size, kv_cache_len)
        softmax_flops, softmax_r_bytes, softmax_w_bytes = self.calculate_softmax_decode(batch_size, kv_cache_len)
        residual_flops, residual_r_bytes, residual_w_bytes = self.calculate_residual_decode(batch_size, kv_cache_len)
        norm_flops, norm_r_bytes, norm_w_bytes = self.calculate_norm_decode(batch_size, kv_cache_len)
        lm_head_flops, lm_head_r_bytes, lm_head_w_bytes = self.calculate_lm_head_decode(batch_size, kv_cache_len)
        generate_flops, generate_r_bytes, generate_w_bytes = self.calculate_generate_decode(batch_size, kv_cache_len)
        kv_cache_store_flops, kv_cache_store_r_bytes, kv_cache_store_w_bytes = self.calculate_kv_cache_store_decode(batch_size, kv_cache_len)

        return DecodingLatencyComponents(
            embed_flops=embed_flops,
            embed_r_bytes=embed_r_bytes,
            embed_w_bytes=embed_w_bytes,
            embed_bytes=embed_r_bytes + embed_w_bytes,

            q_proj_flops=q_proj_flops,
            q_proj_r_bytes=q_proj_r_bytes,
            q_proj_w_bytes=q_proj_w_bytes,
            q_proj_bytes=q_proj_r_bytes + q_proj_w_bytes,

            kv_proj_flops=kv_proj_flops,
            kv_proj_r_bytes=kv_proj_r_bytes,
            kv_proj_w_bytes=kv_proj_w_bytes,
            kv_proj_bytes=kv_proj_r_bytes + kv_proj_w_bytes,

            rope_flops=rope_flops,
            rope_r_bytes=rope_r_bytes,
            rope_w_bytes=rope_w_bytes,
            rope_bytes=rope_r_bytes + rope_w_bytes,

            mlp_flops=mlp_flops,
            mlp_r_bytes=mlp_r_bytes,
            mlp_w_bytes=mlp_w_bytes,
            mlp_bytes=mlp_r_bytes + mlp_w_bytes,

            attention_qk_flops=attention_qk_flops,
            attention_qk_r_bytes=attention_qk_r_bytes,
            attention_qk_w_bytes=attention_qk_w_bytes,
            attention_qk_bytes=attention_qk_r_bytes + attention_qk_w_bytes,

            attention_v_flops=attention_v_flops,
            attention_v_r_bytes=attention_v_r_bytes,
            attention_v_w_bytes=attention_v_w_bytes,
            attention_v_bytes=attention_v_r_bytes + attention_v_w_bytes,

            attention_o_flops=attention_o_flops,
            attention_o_r_bytes=attention_o_r_bytes,
            attention_o_w_bytes=attention_o_w_bytes,
            attention_o_bytes=attention_o_r_bytes + attention_o_w_bytes,

            mask_flops=mask_flops,
            mask_r_bytes=mask_r_bytes,
            mask_w_bytes=mask_w_bytes,
            mask_bytes=mask_r_bytes + mask_w_bytes,

            softmax_flops=softmax_flops,
            softmax_r_bytes=softmax_r_bytes,
            softmax_w_bytes=softmax_w_bytes,
            softmax_bytes=softmax_r_bytes + softmax_w_bytes,

            residual_flops=residual_flops,
            residual_r_bytes=residual_r_bytes,
            residual_w_bytes=residual_w_bytes,
            residual_bytes=residual_r_bytes + residual_w_bytes,

            norm_flops=norm_flops,
            norm_r_bytes=norm_r_bytes,
            norm_w_bytes=norm_w_bytes,
            norm_bytes=norm_r_bytes + norm_w_bytes,

            lm_head_flops=lm_head_flops,
            lm_head_r_bytes=lm_head_r_bytes,
            lm_head_w_bytes=lm_head_w_bytes,
            lm_head_bytes=lm_head_r_bytes + lm_head_w_bytes,

            generate_flops=generate_flops,
            generate_r_bytes=generate_r_bytes,
            generate_w_bytes=generate_w_bytes,
            generate_bytes=generate_r_bytes + generate_w_bytes,

            kv_cache_store_flops=kv_cache_store_flops,
            kv_cache_store_r_bytes=kv_cache_store_r_bytes,
            kv_cache_store_w_bytes=kv_cache_store_w_bytes,
            kv_cache_store_bytes=kv_cache_store_r_bytes + kv_cache_store_w_bytes
        )
    
    def calculate_latency_with_delay_factor(
        self, 
        total_flops: float, 
        total_bytes: float, 
        compute_delay_factor: Optional[float] = None,
        memory_delay_factor: Optional[float] = None
    ) -> float:
        """
        Calculate latency using dynamic delay factors from DelayFactorManager.
        
        Args:
            total_flops: Number of floating point operations
            total_bytes: Number of bytes transferred
            compute_delay_factor: Override compute delay factor (optional)
            memory_delay_factor: Override memory delay factor (optional)
            
        Returns:
            Latency in seconds
        """
        # Get dynamic delay factors if not explicitly provided
        if compute_delay_factor is None or memory_delay_factor is None:
            dynamic_compute_delay, dynamic_memory_delay = self.delay_factor_manager.get_delay_factors(
                flops=total_flops,
                bytes_val=total_bytes
            )
            
            # Use dynamic values if not overridden
            if compute_delay_factor is None:
                compute_delay_factor = dynamic_compute_delay
            if memory_delay_factor is None:
                memory_delay_factor = dynamic_memory_delay
        
        # Calculate actual latencies with delay factors
        compute_time = (total_flops / self.peak_tflops) * compute_delay_factor
        memory_time = (total_bytes / self.memory_bw_bytes) * memory_delay_factor
        
        return max(compute_time, memory_time)
    
    def calculate_tokens_per_second(self, batch_size: int, kv_cache_len: int, num_generated_tokens: int = 100) -> tuple[float, float]:
        """
        Calculate tokens per second and total decoding latency.
        
        Args:
            batch_size: Batch size for inference
            kv_cache_len: Initial KV cache length (input sequence length)
            num_generated_tokens: Number of tokens to generate
            
        Returns:
            Tuple of (tokens_per_second, total_decoding_latency_seconds)
        """
        total_latency = 0.0
        
        for token_idx in range(num_generated_tokens):
            current_kv_cache_len = kv_cache_len + token_idx
            components = self.calculate_all_decode_components(batch_size, current_kv_cache_len)
            
            # Fixed latency operations (계산량이 일정한 연산들)
            embed_time = self.calculate_latency_with_delay_factor(
                components.embed_flops, components.embed_bytes
            )
            q_proj_time = self.calculate_latency_with_delay_factor(
                components.q_proj_flops, components.q_proj_bytes
            )
            kv_proj_time = self.calculate_latency_with_delay_factor(
                components.kv_proj_flops, components.kv_proj_bytes
            )
            mlp_time = self.calculate_latency_with_delay_factor(
                components.mlp_flops, components.mlp_bytes
            )
            attention_o_time = self.calculate_latency_with_delay_factor(
                components.attention_o_flops, components.attention_o_bytes
            )
            residual_time = self.calculate_latency_with_delay_factor(
                components.residual_flops, components.residual_bytes
            )
            norm_time = self.calculate_latency_with_delay_factor(
                components.norm_flops, components.norm_bytes
            )
            kv_cache_store_time = self.calculate_latency_with_delay_factor(
                components.kv_cache_store_flops, components.kv_cache_store_bytes
            )
            
            # Dynamic latency operations (KV 캐시 길이에 따라 변하는 연산들)
            rope_time = self.calculate_latency_with_delay_factor(
                components.rope_flops, components.rope_bytes
            )
            attention_qk_time = self.calculate_latency_with_delay_factor(
                components.attention_qk_flops, components.attention_qk_bytes
            )
            attention_v_time = self.calculate_latency_with_delay_factor(
                components.attention_v_flops, components.attention_v_bytes
            )
            mask_time = self.calculate_latency_with_delay_factor(
                components.mask_flops, components.mask_bytes
            )
            softmax_time = self.calculate_latency_with_delay_factor(
                components.softmax_flops, components.softmax_bytes
            )
            
            # Final operations (layer 밖에서 실행되는 연산들)
            lm_head_time = self.calculate_latency_with_delay_factor(
                components.lm_head_flops, components.lm_head_bytes
            )
            generate_time = self.calculate_latency_with_delay_factor(
                components.generate_flops, components.generate_bytes
            )
            
            # Per-layer operations (executed num_layers times)
            per_layer_latency = (
                q_proj_time + kv_proj_time + rope_time + mlp_time + 
                attention_qk_time + attention_v_time + attention_o_time + 
                mask_time + softmax_time + 2 * residual_time + 2 * norm_time +
                kv_cache_store_time
            )
            
            # Total latency for this token
            token_latency = (
                embed_time +  # Initial embedding
                per_layer_latency * self.model.num_layers +  # All layers
                norm_time +  # Final norm
                lm_head_time +  # LM head projection
                generate_time  # Token generation
            )
            total_latency += token_latency
        
        # Calculate tokens per second
        tokens_per_second = num_generated_tokens / total_latency if total_latency > 0 else 0
        
        return tokens_per_second, total_latency


def main():
    """Example usage of the ForecastingDecodeCalculator."""
    
    # Test parameters
    batch_size = 1
    kv_cache_len = 1944  # Input sequence length
    num_generated_tokens = 512  # Number of tokens to generate
    gpu_config = GPUConfig.L4()
    model_config = ModelConfig.llama_3_1_8B()
    
    # Initialize calculator
    calculator = ForecastingDecodeCalculator(model_config, gpu_config)
    
    # Calculate tokens per second and total decoding latency
    tokens_per_second, total_decoding_latency = calculator.calculate_tokens_per_second(
        batch_size, kv_cache_len, num_generated_tokens
    )
    
    # Print results
    print(f"GPU: {gpu_config.name}")
    print(f"Model: {model_config.name}")
    print(f"Batch Size: {batch_size}")
    print(f"KV Cache Length: {kv_cache_len}")
    print(f"Generated Tokens: {num_generated_tokens}")
    print("\n" + "="*60)
    print("Decoding Performance Results:")
    print("="*60)
    print(f"Tokens per Second: {tokens_per_second:.2f} tok/s")
    print(f"Total Decoding Latency: {total_decoding_latency:.6f}s ({total_decoding_latency*1000:.2f}ms)")
    print(f"Average Latency per Token: {(total_decoding_latency/num_generated_tokens)*1000:.2f}ms/tok")
    print("="*60)
    
    # Test different scenarios
    print("\n" + "="*60)
    print("Performance Analysis for Different Scenarios")
    print("="*60)
    
    scenarios = [
        (1, 512, 50, "Short sequence, 50 tokens"),
        (1, 1024, 100, "Medium sequence, 100 tokens"),
        (1, 2048, 200, "Long sequence, 200 tokens"),
        (1, 4096, 100, "Very long sequence, 100 tokens"),
    ]
    
    for batch, cache_len, gen_tokens, description in scenarios:
        tps, total_lat = calculator.calculate_tokens_per_second(batch, cache_len, gen_tokens)
        avg_lat_per_token = (total_lat / gen_tokens) * 1000
        print(f"{description:30s}: {tps:6.2f} tok/s, {avg_lat_per_token:6.2f}ms/tok")


if __name__ == "__main__":
    main()


def ratio_20():
    print('오차율 20') 