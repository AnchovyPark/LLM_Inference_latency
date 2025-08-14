"""
GPU Specifications Database and Detection System
Provides accurate GPU specifications for theoretical performance calculations
"""

import subprocess
import re
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class GPUSpecs:
    """GPU hardware specifications for performance modeling."""
    name: str
    peak_fp16_tflops: float
    memory_bandwidth_gb_s: float
    sm_count: int
    cuda_cores_per_sm: int
    memory_size_gb: int
    compute_capability: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'peak_fp16_tflops': self.peak_fp16_tflops,
            'memory_bandwidth_gb_s': self.memory_bandwidth_gb_s,
            'sm_count': self.sm_count,
            'cuda_cores_per_sm': self.cuda_cores_per_sm,
            'memory_size_gb': self.memory_size_gb,
            'compute_capability': self.compute_capability
        }


class GPUSpecsDatabase:
    """Database of GPU specifications for accurate performance modeling."""
    
    # Comprehensive GPU specifications database
    GPU_SPECS = {
        # NVIDIA Tesla T4
        'Tesla T4': GPUSpecs(
            name='Tesla T4',
            peak_fp16_tflops=65.0,  # Tensor performance
            memory_bandwidth_gb_s=320.0,  # GDDR6
            sm_count=40,
            cuda_cores_per_sm=64,
            memory_size_gb=16,
            compute_capability='7.5'
        ),
        'T4': GPUSpecs(
            name='T4',
            peak_fp16_tflops=65.0,
            memory_bandwidth_gb_s=320.0,
            sm_count=40,
            cuda_cores_per_sm=64,
            memory_size_gb=16,
            compute_capability='7.5'
        ),
        
        # NVIDIA A10G
        'A10G': GPUSpecs(
            name='A10G',
            peak_fp16_tflops=125.0,  # Tensor performance
            memory_bandwidth_gb_s=600.0,  # GDDR6
            sm_count=80,
            cuda_cores_per_sm=128,
            memory_size_gb=24,
            compute_capability='8.6'
        ),
        'NVIDIA A10G': GPUSpecs(
            name='NVIDIA A10G',
            peak_fp16_tflops=125.0,
            memory_bandwidth_gb_s=600.0,
            sm_count=80,
            cuda_cores_per_sm=128,
            memory_size_gb=24,
            compute_capability='8.6'
        ),
        
        # NVIDIA L40S
        'L40S': GPUSpecs(
            name='L40S',
            peak_fp16_tflops=183.0,  # Tensor performance
            memory_bandwidth_gb_s=864.0,  # GDDR6
            sm_count=144,
            cuda_cores_per_sm=128,
            memory_size_gb=48,
            compute_capability='8.9'
        ),
        'NVIDIA L40S': GPUSpecs(
            name='L40S',
            peak_fp16_tflops=183.0,
            memory_bandwidth_gb_s=864.0,
            sm_count=144,
            cuda_cores_per_sm=128,
            memory_size_gb=48,
            compute_capability='8.9'
        ),
        'L40s': GPUSpecs(  # Handle lowercase 's'
            name='L40s',
            peak_fp16_tflops=183.0,
            memory_bandwidth_gb_s=864.0,
            sm_count=144,
            cuda_cores_per_sm=128,
            memory_size_gb=48,
            compute_capability='8.9'
        ),
        
        # NVIDIA A100 (for reference)
        'A100': GPUSpecs(
            name='A100',
            peak_fp16_tflops=312.0,  # Tensor performance
            memory_bandwidth_gb_s=1600.0,  # HBM2
            sm_count=108,
            cuda_cores_per_sm=64,
            memory_size_gb=40,  # Base model
            compute_capability='8.0'
        ),
        'NVIDIA A100-SXM4-40GB': GPUSpecs(
            name='A100-SXM4-40GB',
            peak_fp16_tflops=312.0,
            memory_bandwidth_gb_s=1600.0,
            sm_count=108,
            cuda_cores_per_sm=64,
            memory_size_gb=40,
            compute_capability='8.0'
        ),
        'NVIDIA A100-SXM4-80GB': GPUSpecs(
            name='A100-SXM4-80GB',
            peak_fp16_tflops=312.0,
            memory_bandwidth_gb_s=1600.0,
            sm_count=108,
            cuda_cores_per_sm=64,
            memory_size_gb=80,
            compute_capability='8.0'
        ),
        
        # NVIDIA RTX 4090 (for reference)
        'RTX 4090': GPUSpecs(
            name='RTX 4090',
            peak_fp16_tflops=166.0,  # Tensor performance
            memory_bandwidth_gb_s=1008.0,  # GDDR6X
            sm_count=128,
            cuda_cores_per_sm=128,
            memory_size_gb=24,
            compute_capability='8.9'
        ),
        'NVIDIA GeForce RTX 4090': GPUSpecs(
            name='RTX 4090',
            peak_fp16_tflops=166.0,
            memory_bandwidth_gb_s=1008.0,
            sm_count=128,
            cuda_cores_per_sm=128,
            memory_size_gb=24,
            compute_capability='8.9'
        ),
        
        # NVIDIA V100 (for reference)
        'V100': GPUSpecs(
            name='V100',
            peak_fp16_tflops=125.0,  # Tensor performance
            memory_bandwidth_gb_s=900.0,  # HBM2
            sm_count=80,
            cuda_cores_per_sm=64,
            memory_size_gb=32,
            compute_capability='7.0'
        ),
        'Tesla V100-SXM2-32GB': GPUSpecs(
            name='V100-SXM2-32GB',
            peak_fp16_tflops=125.0,
            memory_bandwidth_gb_s=900.0,
            sm_count=80,
            cuda_cores_per_sm=64,
            memory_size_gb=32,
            compute_capability='7.0'
        )
    }
    
    @classmethod
    def get_current_gpu_name(cls) -> Optional[str]:
        """Detect current GPU name using nvidia-ml-py or nvidia-smi."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            return name
        except ImportError:
            # Fallback to nvidia-smi if pynvml not available
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, check=True)
                return result.stdout.strip().split('\n')[0]
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None
    
    @classmethod
    def get_specs_by_name(cls, gpu_name: str) -> Optional[GPUSpecs]:
        """Get GPU specifications by name."""
        # Direct lookup
        if gpu_name in cls.GPU_SPECS:
            return cls.GPU_SPECS[gpu_name]
        
        # Fuzzy matching for common variations
        gpu_name_clean = gpu_name.strip()
        
        # Try exact match ignoring case
        for key, specs in cls.GPU_SPECS.items():
            if key.lower() == gpu_name_clean.lower():
                return specs
        
        # Try partial match - prioritize longer (more specific) names
        matching_specs = []
        for key, specs in cls.GPU_SPECS.items():
            if gpu_name_clean.lower() in key.lower() or key.lower() in gpu_name_clean.lower():
                matching_specs.append((key, specs))
        
        # Sort by key length descending (longer names first)
        if matching_specs:
            matching_specs.sort(key=lambda x: len(x[0]), reverse=True)
            return matching_specs[0][1]
        
        return None
    
    @classmethod
    def get_current_gpu_specs(cls) -> Optional[GPUSpecs]:
        """Get specifications for currently available GPU."""
        gpu_name = cls.get_current_gpu_name()
        if gpu_name:
            return cls.get_specs_by_name(gpu_name)
        return None
    
    @classmethod
    def list_supported_gpus(cls) -> Dict[str, GPUSpecs]:
        """List all supported GPU specifications."""
        return cls.GPU_SPECS.copy()
    
    @classmethod
    def infer_gpu_from_filename(cls, filename: str) -> Optional[GPUSpecs]:
        """Infer GPU type from benchmark result filename."""
        filename_lower = filename.lower()
        
        # Common patterns in filenames
        gpu_patterns = {
            't4': 'T4',
            'tesla_t4': 'Tesla T4',
            'a10g': 'A10G', 
            'l40s': 'L40S',
            'l40': 'L40S',  # Handle L40/L40S variations
            'a100': 'A100',
            'v100': 'V100',
            '4090': 'RTX 4090'
        }
        
        for pattern, gpu_key in gpu_patterns.items():
            if pattern in filename_lower:
                return cls.GPU_SPECS.get(gpu_key)
        
        return None


def main():
    """Test GPU specification detection."""
    db = GPUSpecsDatabase()
    
    print("🔍 GPU Specification Detection Test")
    print("=" * 50)
    
    # Test current GPU detection
    current_gpu = db.get_current_gpu_name()
    if current_gpu:
        print(f"Current GPU: {current_gpu}")
        specs = db.get_current_gpu_specs()
        if specs:
            print(f"Specifications found: {specs.name}")
            print(f"  • FP16 Performance: {specs.peak_fp16_tflops} TFLOPS")
            print(f"  • Memory Bandwidth: {specs.memory_bandwidth_gb_s} GB/s")
            print(f"  • SM Count: {specs.sm_count}")
        else:
            print("⚠️  Specifications not found in database")
    else:
        print("⚠️  Could not detect current GPU")
    
    print("\n📋 Supported GPUs:")
    for name, specs in db.list_supported_gpus().items():
        if specs.name not in [s.name for s in [v for k, v in db.list_supported_gpus().items() if k != name]]:
            print(f"  • {specs.name}: {specs.peak_fp16_tflops} TFLOPS, {specs.memory_bandwidth_gb_s} GB/s")
    
    # Test filename inference
    print("\n🔍 Filename Inference Test:")
    test_files = ['T4_DF.json', 'A10G_DF.json', 'L40s_DF.json']
    for filename in test_files:
        specs = db.infer_gpu_from_filename(filename)
        if specs:
            print(f"  • {filename} → {specs.name} ({specs.peak_fp16_tflops} TFLOPS)")
        else:
            print(f"  • {filename} → Not recognized")


if __name__ == "__main__":
    main()