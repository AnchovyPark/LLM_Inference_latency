"""
Forecasting Latency Calculator

This module contains all the latency calculation formulas for LLM inference,
organized as methods within a class for easy use and access.
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
class LatencyComponents:
    """Stores all calculated latency components."""
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





class ForecastingLatencyCalculator:
    """Calculate latency components for LLM inference."""
    
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

    def calculate_embed(self, batch_size:int, seq_len:int) -> tuple[float, float, float]:
        model = self.model

        emded_flops = 0
        embed_r_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element
        embed_w_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element

        return emded_flops, embed_r_bytes, embed_w_bytes

    def calculate_q_projection(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Q projection FLOPs and bytes.
        Kernel fusion이 이뤄지지 않았다고 가정. 각각 읽고, 각각 계산.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        
        q_proj_flops = (
            (model.hidden_size * model.hidden_size)
        ) * batch_size * seq_len * 2
        
        # Read bytes: weight matrix + input
        q_proj_r_bytes = (
            (model.hidden_size * model.hidden_size) +  # Q weight matrix (D,D) read
            (batch_size * seq_len * model.hidden_size)  # Input read
        ) * self.bytes_per_element
        
        # Write bytes: output
        q_proj_w_bytes = (
            (batch_size * model.num_attention_heads * seq_len * model.head_dim)  # Q output write
        ) * self.bytes_per_element
        
        return q_proj_flops, q_proj_r_bytes, q_proj_w_bytes
    
    def calculate_kv_projection(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate KV projection FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        kv_proj_flops = (
            (model.hidden_size * model.head_dim * model.num_key_value_heads) * 2  # K, V
        ) * batch_size * seq_len * 2 
        
        # Read bytes: weight matrices + input
        kv_proj_r_bytes = (
            (model.head_dim * model.num_key_value_heads * model.hidden_size) * 2 +  # K,V weight matrices read
            (batch_size * seq_len * model.hidden_size) * 2  # Input read (for K and V)
        )  * self.bytes_per_element
        
        # Write bytes: output
        kv_proj_w_bytes = (
            (batch_size * model.num_key_value_heads * seq_len * model.head_dim) * 2  # KV output write
        )  * self.bytes_per_element
        
        return kv_proj_flops, kv_proj_r_bytes, kv_proj_w_bytes
    
    def calculate_rope(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Rotary Position Embedding FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        rope_flops = (
            model.num_attention_heads + model.num_key_value_heads
        ) * batch_size * seq_len * model.head_dim * 3 
        
        # Read bytes: Q, K, cos, sin
        rope_r_bytes = (
            ((model.num_attention_heads + model.num_key_value_heads) * batch_size * seq_len * model.head_dim) +  # Q, K read
            (2 * seq_len * model.head_dim)  # cos, sin read
        )  * self.bytes_per_element
        
        # Write bytes: modified Q, K
        rope_w_bytes = (
            (model.num_attention_heads + model.num_key_value_heads) * batch_size * seq_len * model.head_dim
        )  * self.bytes_per_element
        
        return rope_flops, rope_r_bytes, rope_w_bytes
    
    def calculate_mlp(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate MLP (FFN) Projections FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        mlp_flops = (
            2 * (model.hidden_size * model.intermediate_size) * 2 +  # Gate, Up
            2 * (model.intermediate_size * model.hidden_size) +      # Down
            (model.intermediate_size) * 2  # element wise, Act_Fun
        ) * batch_size * seq_len
        
        # Read bytes: input + weight matrices
        mlp_r_bytes = (
            (batch_size * seq_len * model.hidden_size) +  # Read hidden_states
            (model.intermediate_size * model.hidden_size) * 3  # Read Gate, Up, Down weight matrices
        )  * self.bytes_per_element
        
        # Write bytes: output
        mlp_w_bytes = (
            (batch_size * seq_len * model.hidden_size)  # Write hidden_states
        )  * self.bytes_per_element
        
        return mlp_flops, mlp_r_bytes, mlp_w_bytes
    
    def calculate_attention_qk(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Attention Q @ K FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        attention_qk_flops = (
            2 * batch_size * model.num_attention_heads * 
            seq_len * seq_len * model.head_dim 
        )
        
        # Read bytes: Q and K
        attention_qk_r_bytes = (
            (model.num_attention_heads + model.num_key_value_heads) * 
            seq_len * model.head_dim * batch_size  # Read Q and K
        )  * self.bytes_per_element
        
        # Write bytes: attention scores
        attention_qk_w_bytes = (
            model.num_attention_heads * seq_len * seq_len * batch_size  # Write attention scores
        )  * self.bytes_per_element
        
        return attention_qk_flops, attention_qk_r_bytes, attention_qk_w_bytes
    
    def calculate_attention_v(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Attention weights @ V FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        attention_v_flops = (
            2 * batch_size * model.num_attention_heads * 
            seq_len * seq_len * model.head_dim 
        )
        
        # Read bytes: attention weights and V
        attention_v_r_bytes = (
            (seq_len * seq_len * model.num_attention_heads) +  # Read attention weights
            (seq_len * model.head_dim * model.num_key_value_heads)  # Read V
        ) * batch_size  * self.bytes_per_element
        
        # Write bytes: attention output
        attention_v_w_bytes = (
            seq_len * model.head_dim * model.num_attention_heads  # Write attention output
        ) * batch_size  * self.bytes_per_element
        
        return attention_v_flops, attention_v_r_bytes, attention_v_w_bytes
    
    def calculate_attention_o(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Weights_O projection FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        attention_o_flops = (
            (batch_size * seq_len * model.hidden_size**2) * 2
        ) 
        
        # Read bytes: input and weight matrix
        attention_o_r_bytes = (
            (batch_size * seq_len * model.hidden_size) +  # Read input
            (model.hidden_size**2)  # Read O weight matrix
        )  * self.bytes_per_element
        
        # Write bytes: output
        attention_o_w_bytes = (
            (batch_size * seq_len * model.hidden_size)  # Write output
        )  * self.bytes_per_element
        
        return attention_o_flops, attention_o_r_bytes, attention_o_w_bytes
    
    def calculate_attention_mask(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Attention Mask FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        mask_flops = (
            batch_size * model.num_attention_heads * seq_len * seq_len * 3
        ) 
        
        # Read bytes: attention scores and mask
        mask_r_bytes = (
            (batch_size * model.num_attention_heads * seq_len * seq_len) +  # Read attention scores
            (batch_size * seq_len * seq_len)  # Read mask
        )  * self.bytes_per_element
        
        # Write bytes: masked attention scores
        mask_w_bytes = (
            (batch_size * model.num_attention_heads * seq_len * seq_len)  # Write masked scores
        )  * self.bytes_per_element
        
        return mask_flops, mask_r_bytes, mask_w_bytes
    
    def calculate_softmax(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Softmax FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        softmax_flops = (
            batch_size * model.num_attention_heads * seq_len * seq_len * 3
        ) 
        
        # Read bytes: attention scores
        softmax_r_bytes = (
            batch_size * model.num_attention_heads * seq_len * seq_len
        )  * self.bytes_per_element
        
        # Write bytes: normalized attention weights
        softmax_w_bytes = (
            batch_size * model.num_attention_heads * seq_len * seq_len
        )  * self.bytes_per_element
        
        return softmax_flops, softmax_r_bytes, softmax_w_bytes
    
    def calculate_residual(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate Residual Connection FLOPs and bytes.
        Input 값 자체는 다르지만 2번의 잔차연결 모두 행렬차원은 동일하기 때문에 동시에 계산.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        residual_flops = (
            batch_size * seq_len * model.hidden_size * 2
        ) 
        
        # Read bytes: two inputs for each residual connection (2 residual connections per layer)
        residual_r_bytes = (
            batch_size * seq_len * model.hidden_size * 2 * 2  # 2 inputs × 2 residual connections
        )  * self.bytes_per_element
        
        # Write bytes: output of residual connections
        residual_w_bytes = (
            batch_size * seq_len * model.hidden_size * 2  # 2 outputs (one per residual connection)
        )  * self.bytes_per_element
        
        return residual_flops, residual_r_bytes, residual_w_bytes
    
    def calculate_norm(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate RMSNorm FLOPs and bytes.
        수식에 반영할 땐, 'model.num_layers * 2 + 1' 만큼 해주어야 함 (layer당 2번, attention outputs에 1번)
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        norm_flops = (
            batch_size * seq_len * model.hidden_size * 3
        )
        
        # Read bytes: input and norm weights
        norm_r_bytes = (
            (batch_size * seq_len * model.hidden_size) +  # Read input
            model.hidden_size  # Read norm weights
        ) * self.bytes_per_element
        
        # Write bytes: normalized output
        norm_w_bytes = (
            (batch_size * seq_len * model.hidden_size)  # Write normalized output
        ) * self.bytes_per_element
        
        return norm_flops, norm_r_bytes, norm_w_bytes
    
    def calculate_lm_head(self, batch_size: int, seq_len: int) -> tuple[float, float, float]:
        model = self.model
        """
        Calculate LM Head Projection FLOPs and bytes.
        
        Returns:
            tuple: (flops, r_bytes, w_bytes)
        """
        lm_head_flops = batch_size * seq_len * model.hidden_size * model.vocab_size * 2
        
        # Read bytes: input and weight matrix
        lm_head_r_bytes = (
            batch_size * seq_len * model.hidden_size +  # Read input
            model.hidden_size * model.vocab_size  # Read weight matrix
        ) * self.bytes_per_element
        
        # Write bytes: logits output
        lm_head_w_bytes = (
            batch_size * seq_len * model.vocab_size  # Write logits
        ) * self.bytes_per_element
        
        return lm_head_flops, lm_head_r_bytes, lm_head_w_bytes
    
    def calculate_generate(self, batch_size:int, seq_len:int) -> tuple[float, float, float]:
        model = self.model

        generate_flops = 0
        generate_r_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element
        generate_w_bytes = batch_size * seq_len * model.hidden_size * self.bytes_per_element

        return generate_flops, generate_r_bytes, generate_w_bytes
        
    def calculate_all_components(self, batch_size: int, seq_len: int) -> LatencyComponents:
        """
        Calculate all latency components at once.
        
        Args:
            batch_size: Batch size for inference
            seq_len: Sequence length (input length)
            
        Returns:
            LatencyComponents object with all calculated values
        """
        embed_flops, embed_r_bytes, embed_w_bytes = self.calculate_embed(batch_size, seq_len)
        q_proj_flops, q_proj_r_bytes, q_proj_w_bytes = self.calculate_q_projection(batch_size, seq_len)
        kv_proj_flops, kv_proj_r_bytes, kv_proj_w_bytes = self.calculate_kv_projection(batch_size, seq_len)
        rope_flops, rope_r_bytes, rope_w_bytes = self.calculate_rope(batch_size, seq_len)
        mlp_flops, mlp_r_bytes, mlp_w_bytes = self.calculate_mlp(batch_size, seq_len)
        attention_qk_flops, attention_qk_r_bytes, attention_qk_w_bytes = self.calculate_attention_qk(batch_size, seq_len)
        attention_v_flops, attention_v_r_bytes, attention_v_w_bytes = self.calculate_attention_v(batch_size, seq_len)
        attention_o_flops, attention_o_r_bytes, attention_o_w_bytes = self.calculate_attention_o(batch_size, seq_len)
        mask_flops, mask_r_bytes, mask_w_bytes = self.calculate_attention_mask(batch_size, seq_len)
        softmax_flops, softmax_r_bytes, softmax_w_bytes = self.calculate_softmax(batch_size, seq_len)
        residual_flops, residual_r_bytes, residual_w_bytes = self.calculate_residual(batch_size, seq_len)
        norm_flops, norm_r_bytes, norm_w_bytes = self.calculate_norm(batch_size, seq_len)
        lm_head_flops, lm_head_r_bytes, lm_head_w_bytes = self.calculate_lm_head(batch_size, seq_len)
        generate_flops, generate_r_bytes, generate_w_bytes = self.calculate_generate(batch_size, seq_len)

        return LatencyComponents(

            embed_flops= embed_flops,
            embed_r_bytes= embed_r_bytes,
            embed_w_bytes= embed_w_bytes,
            embed_bytes= embed_w_bytes+ embed_r_bytes,

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
            generate_bytes=generate_r_bytes+ generate_w_bytes


        )
    
    def calculate_latency_with_delay_factor(
        self, 
        total_flops: float, 
        total_bytes: float, 
        compute_delay_factor: Optional[float] = None,
        memory_delay_factor: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Calculate latency using dynamic delay factors from DelayFactorManager.
        
        Args:
            total_flops: Number of floating point operations
            total_bytes: Number of bytes transferred
            compute_delay_factor: Override compute delay factor (optional)
            memory_delay_factor: Override memory delay factor (optional)
            
        Returns:
            Tuple of (compute_time, memory_time) in seconds
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
        
        return max( compute_time, memory_time)
    
    def mix_delay_factor_components(self, batch_size: int, seq_len: int) -> float:
        """
        Calculate total latency by getting components and summing max latency of each operator.
        
        Args:
            batch_size: Batch size for inference
            seq_len: Sequence length (input length)
            
        Returns:
            Total latency in seconds
        """
        components = self.calculate_all_components(batch_size, seq_len)
        
        total_latency = 0.0
        
        # Embedding
        embed_time= self.calculate_latency_with_delay_factor(
            components.embed_flops,
            components.embed_bytes
        )

        # Q Projection
        q_proj_time = self.calculate_latency_with_delay_factor(
            components.q_proj_flops,
            components.q_proj_bytes
        )
        
        # KV Projection  
        kv_proj_time = self.calculate_latency_with_delay_factor(
            components.kv_proj_flops,
            components.kv_proj_bytes
        )
        
        # RoPE
        rope_time = self.calculate_latency_with_delay_factor(
            components.rope_flops,
            components.rope_bytes
        )
        
        # MLP
        mlp_time = self.calculate_latency_with_delay_factor(
            components.mlp_flops,
            components.mlp_bytes
        )
        
        # Attention Q@K
        attention_qk_time = self.calculate_latency_with_delay_factor(
            components.attention_qk_flops,
            components.attention_qk_bytes
        )
        
        # Attention @V
        attention_v_time = self.calculate_latency_with_delay_factor(
            components.attention_v_flops,
            components.attention_v_bytes
        )
        
        # Attention O
        attention_o_time = self.calculate_latency_with_delay_factor(
            components.attention_o_flops,
            components.attention_o_bytes
        )
        
        # Attention Mask
        mask_time = self.calculate_latency_with_delay_factor(
            components.mask_flops,
            components.mask_bytes
        )
        
        # Softmax
        softmax_time = self.calculate_latency_with_delay_factor(
            components.softmax_flops,
            components.softmax_bytes
        )
        
        # Residual Connection
        residual_time = self.calculate_latency_with_delay_factor(
            components.residual_flops,
            components.residual_bytes
        )
        
        # Layer Normalization
        norm_time = self.calculate_latency_with_delay_factor(
            components.norm_flops,
            components.norm_bytes
        )
        
        # LM Head
        lm_head_time = self.calculate_latency_with_delay_factor(
            components.lm_head_flops,
            components.lm_head_bytes
        )

        # Generate (Logits -> Token ID )
        generate_time = self.calculate_latency_with_delay_factor(
            components.generate_flops,
            components.generate_bytes
        )

        total_time = (
            embed_time +     
            (rope_time + mlp_time + attention_qk_time + q_proj_time + kv_proj_time + attention_o_time + attention_v_time + softmax_time + mask_time + 2*residual_time) * self.model.num_layers + 
            lm_head_time  + generate_time + norm_time * (2* self.model.num_layers + 1)
            )
        
        return {
            "embed_time" : embed_time,
            "q_proj_time" : q_proj_time,
            "kv_proj_time" : kv_proj_time,
            "rope_time" : rope_time,
            "mlp_time" : mlp_time, 
            "attention_qk_time" : attention_qk_time,
            "lm_head_time" : lm_head_time,
            "norm_time" : norm_time,
            "residual_time" : residual_time,
            "softmax_time" : softmax_time,
            "mask_time" : mask_time,
            "attention_o_time" : attention_o_time,
            "attention_v_time" : attention_v_time,
            "generate_time" : generate_time,
            "total_time" : total_time
        }


        


def main():
    """Example usage of the ForecastingLatencyCalculator using mix_delay_factor_components."""
    
    # Test parameters (hardcoded)
    batch_size = 1
    seq_len = 1944
    gpu_config = GPUConfig.L4()
    model_config = ModelConfig.llama_3_1_8B()  # Example model
    
    # Initialize calculator
    calculator = ForecastingLatencyCalculator(model_config, gpu_config)
    final_latencies = calculator.mix_delay_factor_components(batch_size=batch_size, seq_len=seq_len)

    # Print results
    print(f"GPU: {gpu_config.name}")
    print(f"Model: {model_config.name}")
    print(f"Batch Size: {batch_size}, Sequence Length: {seq_len}")
    print("\n" + "="*60)
    print("Component Latencies (in seconds):")
    print("="*60)
    
    # Print each component latency directly from mix_delay_factor_components return values
    print(f"Embed               : {final_latencies['embed_time']:.6f}s ({final_latencies['embed_time']*1000:.2f}ms)")
    print(f"Q Proj              : {final_latencies['q_proj_time']:.6f}s ({final_latencies['q_proj_time']*1000:.2f}ms)")
    print(f"Kv Proj             : {final_latencies['kv_proj_time']:.6f}s ({final_latencies['kv_proj_time']*1000:.2f}ms)")
    print(f"Rope                : {final_latencies['rope_time']:.6f}s ({final_latencies['rope_time']*1000:.2f}ms)")
    print(f"Mlp                 : {final_latencies['mlp_time']:.6f}s ({final_latencies['mlp_time']*1000:.2f}ms)")
    print(f"Attention Qk        : {final_latencies['attention_qk_time']:.6f}s ({final_latencies['attention_qk_time']*1000:.2f}ms)")
    print(f"Lm Head             : {final_latencies['lm_head_time']:.6f}s ({final_latencies['lm_head_time']*1000:.2f}ms)")
    print(f"Norm                : {final_latencies['norm_time']:.6f}s ({final_latencies['norm_time']*1000:.2f}ms)")
    print(f"Residual            : {final_latencies['residual_time']:.6f}s ({final_latencies['residual_time']*1000:.2f}ms)")
    print(f"Softmax             : {final_latencies['softmax_time']:.6f}s ({final_latencies['softmax_time']*1000:.2f}ms)")
    print(f"Mask                : {final_latencies['mask_time']:.6f}s ({final_latencies['mask_time']*1000:.2f}ms)")
    print(f"Attention O         : {final_latencies['attention_o_time']:.6f}s ({final_latencies['attention_o_time']*1000:.2f}ms)")
    print(f"Attention V         : {final_latencies['attention_v_time']:.6f}s ({final_latencies['attention_v_time']*1000:.2f}ms)")
    print(f"Generate            : {final_latencies['generate_time']:.6f}s ({final_latencies['generate_time']*1000:.2f}ms)")
    
    print("\n" + "="*60)
    print(f"TOTAL LATENCY: {final_latencies['total_time']:.6f}s ({final_latencies['total_time']*1000:.2f}ms)")
    print("="*60)
    
    # Additional analysis with different scenarios
    print("\n" + "="*60)
    print("Latency Analysis for Different Scenarios")
    print("="*60)
    
    scenarios = [
        (1, 512, "Short sequence"),
        (1, 1024, "Medium sequence"), 
        (1, 2048, "Long sequence"),
        (1, 4096, "Very long sequence"),
    ]
    
    for batch, seq, description in scenarios:
        result = calculator.mix_delay_factor_components(batch, seq)
        # Use the total_time directly from mix_delay_factor_components
        total_time = result['total_time']
        print(f"{description:20s} (batch={batch}, seq={seq:4d}): {total_time:.6f}s ({total_time*1000:.2f}ms)")
    
    return final_latencies


if __name__ == "__main__":
    main()