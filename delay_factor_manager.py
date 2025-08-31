"""
Delay Factor Manager

This module manages hardware-specific delay factors based on FLOPs and memory bytes.
It loads delay factors from JSON files and provides interpolated values for given operations.
"""

import json
import numpy as np
from typing import Dict, Tuple, Optional, Union
from pathlib import Path


class DelayFactorManager:
    """
    Manages delay factors for different GPUs based on FLOPs and memory bytes.
    Provides interpolation for values not explicitly defined in the configuration.
    """
    
    def __init__(self, gpu_name: str, config_file: str = None):
        """
        Initialize the delay factor manager for a specific GPU.
        
        Args:
            gpu_name: Name of the GPU (e.g., 'A100', 'H100', 'L4', 'T4')
            config_file: Path to the JSON configuration file containing delay factors
        """
        self.gpu_name = gpu_name
        self.config_file = config_file or "/Users/anchovy-mac/Desktop/calculating/data/delay_factors.json"
        self.flops_delay_factors = {}
        self.memory_delay_factors = {}
        self._load_delay_factors()
    
    def _load_delay_factors(self):
        """Load delay factors from the JSON configuration file."""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
            
            if self.gpu_name not in data:
                print(f"Warning: GPU '{self.gpu_name}' not found in config. Using default values.")
                self._set_default_factors()
                return
            
            gpu_data = data[self.gpu_name]
            
            # Convert string keys to float for FLOPs delay factors
            if 'flops_delay_factors' in gpu_data:
                self.flops_delay_factors = {
                    float(k): float(v) 
                    for k, v in gpu_data['flops_delay_factors'].items() 
                    if k != '...'
                }
            
            # Convert string keys to float for memory delay factors
            if 'memory_delay_factors' in gpu_data:
                self.memory_delay_factors = {
                    float(k): float(v) 
                    for k, v in gpu_data['memory_delay_factors'].items() 
                    if k != '...'
                }
            
            print(f"Loaded delay factors for GPU: {self.gpu_name}")
            print(f"  - FLOPs points: {len(self.flops_delay_factors)}")
            print(f"  - Memory points: {len(self.memory_delay_factors)}")
            
        except FileNotFoundError:
            print(f"Config file not found: {self.config_file}. Using default values.")
            self._set_default_factors()
        except Exception as e:
            print(f"Error loading delay factors: {e}. Using default values.")
            self._set_default_factors()
    
    def _set_default_factors(self):
        """Set default delay factors for GPUs not in the configuration."""
        # Default factors - all set to 1.0 (no delay, theoretical performance)
        default_configs = {
            'T4': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'L4': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'A10G': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'L40': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'L40S': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'A100': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'A100-40GB': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'A100-80GB': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'H100': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'H100-PCIe': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            },
            'H100-SXM': {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            }
        }
        
        # Use specific GPU defaults if available, otherwise use generic defaults
        if self.gpu_name in default_configs:
            config = default_configs[self.gpu_name]
        else:
            # Generic default for unknown GPUs - all 1.0
            config = {
                'flops': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)],
                'memory': [(1e3, 1.0), (1e6, 1.0), (1e9, 1.0), (1e12, 1.0)]
            }
        
        self.flops_delay_factors = dict(config['flops'])
        self.memory_delay_factors = dict(config['memory'])
    
    def get_flops_delay_factor(self, flops: float) -> float:
        """
        Get the delay factor for a given number of FLOPs.
        Uses logarithmic interpolation for values between defined points.
        
        Args:
            flops: Number of floating-point operations
            
        Returns:
            Interpolated delay factor
        """
        return self._interpolate_delay_factor(flops, self.flops_delay_factors)
    
    def get_memory_delay_factor(self, bytes_val: float) -> float:
        """
        Get the delay factor for a given number of memory bytes.
        Uses logarithmic interpolation for values between defined points.
        
        Args:
            bytes_val: Number of bytes to transfer
            
        Returns:
            Interpolated delay factor
        """
        return self._interpolate_delay_factor(bytes_val, self.memory_delay_factors)
    
    def _interpolate_delay_factor(self, value: float, factors_dict: Dict[float, float]) -> float:
        """
        Interpolate delay factor using logarithmic interpolation.
        
        Args:
            value: The value (FLOPs or bytes) to interpolate for
            factors_dict: Dictionary mapping values to delay factors
            
        Returns:
            Interpolated delay factor
        """
        if not factors_dict:
            return 1.0  # Default delay factor (theoretical performance)
        
        # Get sorted keys
        sorted_keys = sorted(factors_dict.keys())
        
        # Handle edge cases
        if value <= sorted_keys[0]:
            return factors_dict[sorted_keys[0]]
        if value >= sorted_keys[-1]:
            return factors_dict[sorted_keys[-1]]
        
        # Find surrounding points for interpolation
        for i in range(len(sorted_keys) - 1):
            if sorted_keys[i] <= value <= sorted_keys[i + 1]:
                x1, x2 = sorted_keys[i], sorted_keys[i + 1]
                y1, y2 = factors_dict[x1], factors_dict[x2]
                
                # Logarithmic interpolation
                if x1 > 0 and x2 > 0 and value > 0:
                    log_x1, log_x2, log_value = np.log10(x1), np.log10(x2), np.log10(value)
                    weight = (log_value - log_x1) / (log_x2 - log_x1)
                    return y1 + weight * (y2 - y1)
                else:
                    # Linear interpolation as fallback
                    weight = (value - x1) / (x2 - x1)
                    return y1 + weight * (y2 - y1)
        
        # Should not reach here, but return default
        return 1.0
    
    def get_delay_factors(self, flops: float, bytes_val: float) -> Tuple[float, float]:
        """
        Get both compute and memory delay factors for an operation.
        
        Args:
            flops: Number of floating-point operations
            bytes_val: Number of bytes to transfer
            
        Returns:
            Tuple of (compute_delay_factor, memory_delay_factor)
        """
        compute_delay = self.get_flops_delay_factor(flops)
        memory_delay = self.get_memory_delay_factor(bytes_val)
        return compute_delay, memory_delay
    
    def get_effective_delay_factor(self, flops: float, bytes_val: float, 
                                  compute_time: float, memory_time: float) -> float:
        """
        Calculate the effective delay factor based on bottleneck analysis.
        
        Args:
            flops: Number of floating-point operations
            bytes_val: Number of bytes to transfer
            compute_time: Theoretical compute time
            memory_time: Theoretical memory time
            
        Returns:
            Effective delay factor for the operation
        """
        compute_delay, memory_delay = self.get_delay_factors(flops, bytes_val)
        
        # Determine bottleneck and return appropriate delay factor
        if compute_time > memory_time:
            # Compute-bound
            return compute_delay
        else:
            # Memory-bound
            return memory_delay
    
    def save_delay_factors(self, output_file: str = None):
        """
        Save current delay factors to a JSON file.
        
        Args:
            output_file: Path to save the delay factors (optional)
        """
        output_file = output_file or self.config_file
        
        # Load existing data or create new
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
        except:
            data = {}
        
        # Update with current GPU's factors
        data[self.gpu_name] = {
            'flops_delay_factors': {str(k): v for k, v in self.flops_delay_factors.items()},
            'memory_delay_factors': {str(k): v for k, v in self.memory_delay_factors.items()}
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved delay factors for {self.gpu_name} to {output_file}")
    
    def update_delay_factor(self, factor_type: str, value: float, delay_factor: float):
        """
        Update a specific delay factor.
        
        Args:
            factor_type: 'flops' or 'memory'
            value: The FLOPs or bytes value
            delay_factor: The new delay factor
        """
        if factor_type == 'flops':
            self.flops_delay_factors[value] = delay_factor
        elif factor_type == 'memory':
            self.memory_delay_factors[value] = delay_factor
        else:
            raise ValueError(f"Invalid factor_type: {factor_type}. Must be 'flops' or 'memory'")
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the current delay factors.
        
        Returns:
            Dictionary containing GPU name and delay factor ranges
        """
        flops_values = list(self.flops_delay_factors.values()) if self.flops_delay_factors else [2.0]
        memory_values = list(self.memory_delay_factors.values()) if self.memory_delay_factors else [2.0]
        
        return {
            'gpu_name': self.gpu_name,
            'flops_delay_range': (min(flops_values), max(flops_values)),
            'memory_delay_range': (min(memory_values), max(memory_values)),
            'num_flops_points': len(self.flops_delay_factors),
            'num_memory_points': len(self.memory_delay_factors)
        }


def main():
    """Example usage of the DelayFactorManager."""
    
    print("="*80)
    print("Delay Factor Manager - Example Usage")
    print("="*80)
    
    # Example 1: Initialize manager for A100 GPU
    print("\n1. Initialize for A100 GPU")
    print("-" * 40)
    manager = DelayFactorManager('A100', '/Users/anchovy-mac/Desktop/calculating/data/example.json')
    print(manager.get_summary())
    
    # Example 2: Get delay factors for specific operations
    print("\n2. Get delay factors for operations")
    print("-" * 40)
    
    test_cases = [
        (1e3, 1e4),    # Small operation
        (1e6, 1e7),    # Medium operation
        (1e9, 1e10),   # Large operation
        (1e12, 1e13),  # Very large operation
    ]
    
    for flops, bytes_val in test_cases:
        compute_delay = manager.get_flops_delay_factor(flops)
        memory_delay = manager.get_memory_delay_factor(bytes_val)
        print(f"FLOPs: {flops:.0e}, Bytes: {bytes_val:.0e}")
        print(f"  Compute delay factor: {compute_delay:.2f}")
        print(f"  Memory delay factor: {memory_delay:.2f}")
    
    # Example 3: Effective delay factor based on bottleneck
    print("\n3. Effective delay factor (bottleneck-aware)")
    print("-" * 40)
    
    flops = 1e9
    bytes_val = 1e8
    compute_time = 0.001  # 1ms
    memory_time = 0.002   # 2ms
    
    effective_delay = manager.get_effective_delay_factor(
        flops, bytes_val, compute_time, memory_time
    )
    print(f"Operation: FLOPs={flops:.0e}, Bytes={bytes_val:.0e}")
    print(f"Compute time: {compute_time:.3f}s, Memory time: {memory_time:.3f}s")
    print(f"Bottleneck: {'Memory' if memory_time > compute_time else 'Compute'}")
    print(f"Effective delay factor: {effective_delay:.2f}")
    
    # Example 4: Test with different GPUs
    print("\n4. Compare different GPUs")
    print("-" * 40)
    
    gpus = ['T4', 'L4', 'A100', 'H100']
    flops = 1e10
    bytes_val = 1e9
    
    for gpu_name in gpus:
        gpu_manager = DelayFactorManager(gpu_name)
        compute_delay, memory_delay = gpu_manager.get_delay_factors(flops, bytes_val)
        print(f"{gpu_name:10s} - Compute: {compute_delay:.2f}, Memory: {memory_delay:.2f}")
    
    print("\n" + "="*80)
    print("Delay Factor Manager Ready")
    print("="*80)


if __name__ == "__main__":
    main()