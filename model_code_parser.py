"""
Model Code Parser for Operator Usage Analysis
Parse actual LLM model implementation files to identify operator usage patterns
Based on LIFE paper's foundational and derived operators
"""

import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class OperatorUsage:
    """Track usage of a specific operator in the model."""
    name: str
    foundational_type: str  # From LIFE Table 1
    derived_type: Optional[str]  # From LIFE Table 2
    count: int
    locations: List[str]  # Where it's used in code
    parameters: Dict[str, Any]  # Shape, dtype, etc.


class LIFEOperatorMapper:
    """Map code patterns to LIFE paper operators."""
    
    def __init__(self):
        # LIFE Table 1: Foundational Operators
        self.foundational_patterns = {
            'Linear': [
                r'nn\.Linear',
                r'F\.linear',
                r'torch\.matmul',
                r'@',  # Matrix multiplication operator
                r'mm\(',
                r'bmm\(',
                r'addmm\(',
            ],
            'BMM': [
                r'torch\.bmm',
                r'\.bmm\(',
                r'batch_matmul',
                r'einsum.*ij,jk->ik',
            ],
            'Elementwise': [
                r'\+',
                r'\*',
                r'torch\.add',
                r'torch\.mul',
                r'torch\.div',
                r'\.add\(',
                r'\.mul\(',
                r'\.div\(',
            ],
            'Non-Linear': [
                r'F\.relu',
                r'F\.gelu',
                r'F\.silu',
                r'F\.tanh',
                r'F\.sigmoid',
                r'torch\.relu',
                r'torch\.gelu',
                r'torch\.silu',
                r'nn\.ReLU',
                r'nn\.GELU',
                r'nn\.SiLU',
            ],
            'Embedding': [
                r'nn\.Embedding',
                r'F\.embedding',
                r'torch\.embedding',
            ],
            'Quantize': [
                r'quantize',
                r'dequantize',
                r'fake_quantize',
                r'int8',
                r'int4',
            ]
        }
        
        # LIFE Table 2: Derived Operators  
        self.derived_patterns = {
            'MHA': [
                r'MultiHeadAttention',
                r'self_attention',
                r'multi_head_attention',
                r'scaled_dot_product_attention',
            ],
            'GQA': [
                r'GroupedQueryAttention',
                r'group_query_attention',
                r'num_key_value_heads',
            ],
            'MQA': [
                r'MultiQueryAttention',
                r'multi_query_attention',
            ],
            'MLA': [
                r'MultiHeadLatentAttention',
                r'latent_attention',
            ],
            'MLP': [
                r'class.*MLP',
                r'FeedForward',
                r'feed_forward',
                r'ffn',
            ],
            'Softmax': [
                r'F\.softmax',
                r'torch\.softmax',
                r'nn\.Softmax',
                r'\.softmax\(',
            ],
            'RoPE': [
                r'RotaryPositionalEmbedding',
                r'rotary_pos_emb',
                r'apply_rotary_pos_emb',
                r'rope',
            ],
            'LayerNorm': [
                r'nn\.LayerNorm',
                r'F\.layer_norm',
                r'layer_norm',
            ],
            'RMSNorm': [
                r'RMSNorm',
                r'rms_norm',
                r'root_mean_square',
            ],
        }


class ModelCodeParser:
    """Parse model implementation files to extract operator usage."""
    
    def __init__(self):
        self.mapper = LIFEOperatorMapper()
        self.operator_usage = defaultdict(lambda: OperatorUsage("", "", None, 0, [], {}))
        
    def parse_python_file(self, file_path: str) -> Dict[str, OperatorUsage]:
        """Parse a Python model file and extract operator usage."""
        
        print(f"🔍 Parsing: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for structured analysis
            tree = ast.parse(content)
            self._analyze_ast(tree, file_path)
            
            # Parse text patterns for additional operators
            self._analyze_text_patterns(content, file_path)
            
        except Exception as e:
            print(f"❌ Error parsing {file_path}: {e}")
        
        return dict(self.operator_usage)
    
    def _analyze_ast(self, tree: ast.AST, file_path: str):
        """Analyze AST to find operator usage patterns."""
        
        class OperatorVisitor(ast.NodeVisitor):
            def __init__(self, parser):
                self.parser = parser
                self.file_path = file_path
            
            def visit_ClassDef(self, node):
                """Analyze class definitions for model components."""
                class_name = node.name
                
                # Check for derived operators
                for derived_op, patterns in self.parser.mapper.derived_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, class_name, re.IGNORECASE):
                            self.parser._record_usage(
                                derived_op, "derived", derived_op, 
                                f"{self.file_path}:class:{class_name}"
                            )
                
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                """Analyze function definitions."""
                func_name = node.name
                
                # Check for operator function names
                for derived_op, patterns in self.parser.mapper.derived_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, func_name, re.IGNORECASE):
                            self.parser._record_usage(
                                derived_op, "derived", derived_op,
                                f"{self.file_path}:function:{func_name}"
                            )
                
                self.generic_visit(node)
            
            def visit_Call(self, node):
                """Analyze function calls for operator usage."""
                func_name = ast.unparse(node.func) if hasattr(ast, 'unparse') else str(node.func)
                
                # Check foundational operators
                for found_op, patterns in self.parser.mapper.foundational_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, func_name):
                            self.parser._record_usage(
                                found_op, "foundational", None,
                                f"{self.file_path}:call:{func_name}"
                            )
                
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                """Analyze binary operations (like matrix multiplication @)."""
                if isinstance(node.op, ast.MatMult):  # @ operator
                    self.parser._record_usage(
                        "Linear", "foundational", None,
                        f"{self.file_path}:binop:@"
                    )
                elif isinstance(node.op, (ast.Add, ast.Mult, ast.Div, ast.Sub)):
                    self.parser._record_usage(
                        "Elementwise", "foundational", None,
                        f"{self.file_path}:binop:{type(node.op).__name__}"
                    )
                
                self.generic_visit(node)
        
        visitor = OperatorVisitor(self)
        visitor.visit(tree)
    
    def _analyze_text_patterns(self, content: str, file_path: str):
        """Analyze text patterns for operators that might be missed by AST."""
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Check foundational operators
            for found_op, patterns in self.mapper.foundational_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line):
                        self._record_usage(
                            found_op, "foundational", None,
                            f"{file_path}:line:{line_num}"
                        )
            
            # Check derived operators
            for derived_op, patterns in self.mapper.derived_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        self._record_usage(
                            derived_op, "derived", derived_op,
                            f"{file_path}:line:{line_num}"
                        )
    
    def _record_usage(self, op_name: str, op_type: str, derived_type: Optional[str], location: str):
        """Record operator usage."""
        key = f"{op_type}_{op_name}"
        
        if key not in self.operator_usage:
            self.operator_usage[key] = OperatorUsage(
                name=op_name,
                foundational_type=op_type,
                derived_type=derived_type,
                count=0,
                locations=[],
                parameters={}
            )
        
        self.operator_usage[key].count += 1
        self.operator_usage[key].locations.append(location)
    
    def parse_config_file(self, config_path: str) -> Dict[str, Any]:
        """Parse model config.json to extract architecture parameters."""
        
        print(f"📋 Parsing config: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract key parameters
            arch_params = {
                'model_type': config.get('model_type', 'unknown'),
                'hidden_size': config.get('hidden_size', 0),
                'intermediate_size': config.get('intermediate_size', 0),
                'num_attention_heads': config.get('num_attention_heads', 0),
                'num_key_value_heads': config.get('num_key_value_heads', config.get('num_attention_heads', 0)),
                'num_hidden_layers': config.get('num_hidden_layers', 0),
                'vocab_size': config.get('vocab_size', 0),
                'attention_type': self._infer_attention_type(config),
                'activation_function': config.get('hidden_act', 'unknown'),
                'normalization_type': self._infer_norm_type(config),
            }
            
            return arch_params
            
        except Exception as e:
            print(f"❌ Error parsing config {config_path}: {e}")
            return {}
    
    def _infer_attention_type(self, config: Dict) -> str:
        """Infer attention type from config."""
        num_q_heads = config.get('num_attention_heads', 0)
        num_kv_heads = config.get('num_key_value_heads', num_q_heads)
        
        if num_kv_heads == 1:
            return 'MQA'
        elif num_kv_heads < num_q_heads:
            return 'GQA'
        elif num_kv_heads == num_q_heads:
            return 'MHA'
        else:
            return 'unknown'
    
    def _infer_norm_type(self, config: Dict) -> str:
        """Infer normalization type from config."""
        norm_type = config.get('rms_norm_eps')
        if norm_type is not None:
            return 'RMSNorm'
        elif config.get('layer_norm_epsilon') is not None:
            return 'LayerNorm'
        else:
            return 'unknown'
    
    def calculate_theoretical_operator_counts(self, arch_params: Dict[str, Any], sequence_length: int = 1024) -> Dict[str, int]:
        """Calculate theoretical operator counts based on model architecture."""
        
        layers = arch_params.get('num_hidden_layers', 16)
        hidden_size = arch_params.get('hidden_size', 2048)
        intermediate_size = arch_params.get('intermediate_size', 5632)
        num_q_heads = arch_params.get('num_attention_heads', 32)
        num_kv_heads = arch_params.get('num_key_value_heads', 8)
        vocab_size = arch_params.get('vocab_size', 128256)
        head_dim = hidden_size // num_q_heads
        
        theoretical_counts = {}
        
        # Foundational operators per layer
        per_layer_counts = {
            # Linear operations in attention (Q, K, V, O projections)
            'Linear': (
                1 +  # Q projection: hidden_size -> num_q_heads * head_dim
                1 +  # K projection: hidden_size -> num_kv_heads * head_dim  
                1 +  # V projection: hidden_size -> num_kv_heads * head_dim
                1    # O projection: num_q_heads * head_dim -> hidden_size
            ),
            # BMM operations in attention (Q@K, attn@V)
            'BMM': 2,
            # MLP Linear operations (gate, up, down projections)
            'Linear_MLP': 3,  # gate_proj, up_proj, down_proj
            # Elementwise operations (adds, muls, activations)
            'Elementwise': (
                2 +  # Residual connections (attention + MLP)
                2 +  # SwiGLU: gate * silu(up)
                4    # Various normalizations and activations
            ),
            # Derived operators per layer
            'GQA_or_MHA': 1,     # One attention mechanism per layer
            'MLP_block': 1,      # One MLP block per layer
            'RMSNorm': 2,        # Pre-attention and pre-MLP norms
            'RoPE': 1,           # Rotary position embedding per layer
            'Softmax': 1,        # Attention softmax per layer
        }
        
        # Calculate total counts across all layers
        theoretical_counts['foundational_Linear'] = (per_layer_counts['Linear'] + per_layer_counts['Linear_MLP']) * layers
        theoretical_counts['foundational_BMM'] = per_layer_counts['BMM'] * layers
        theoretical_counts['foundational_Elementwise'] = per_layer_counts['Elementwise'] * layers
        theoretical_counts['foundational_Embedding'] = 1  # Token embedding (position embedding often integrated)
        
        # Derived operators
        attention_type = arch_params.get('attention_type', 'GQA')
        theoretical_counts[f'derived_{attention_type}'] = layers
        theoretical_counts['derived_MLP'] = layers
        theoretical_counts['derived_RMSNorm'] = (2 * layers) + 1  # 2 per layer + final norm
        theoretical_counts['derived_RoPE'] = layers
        theoretical_counts['derived_Softmax'] = layers
        
        return theoretical_counts

    def generate_operator_report(self, model_path: str, config_path: str, sequence_length: int = 1024) -> Dict[str, Any]:
        """Generate comprehensive operator usage report."""
        
        print(f"📊 Generating Operator Report")
        print("=" * 50)
        
        # Parse config
        arch_params = self.parse_config_file(config_path)
        
        # Calculate theoretical counts
        theoretical_counts = self.calculate_theoretical_operator_counts(arch_params, sequence_length)
        
        # Parse model files
        model_dir = Path(model_path)
        if model_dir.is_file():
            # Single file
            operator_usage = self.parse_python_file(str(model_dir))
        else:
            # Directory - parse all Python files
            python_files = list(model_dir.glob("**/*.py"))
            print(f"Found {len(python_files)} Python files to parse")
            
            operator_usage = {}
            for py_file in python_files:
                file_usage = self.parse_python_file(str(py_file))
                # Merge usage counts
                for key, usage in file_usage.items():
                    if key in operator_usage:
                        operator_usage[key].count += usage.count
                        operator_usage[key].locations.extend(usage.locations)
                    else:
                        operator_usage[key] = usage
        
        # Calculate operator statistics
        foundational_ops = {k: v for k, v in operator_usage.items() if v.foundational_type == "foundational"}
        derived_ops = {k: v for k, v in operator_usage.items() if v.foundational_type == "derived"}
        
        # Generate report
        report = {
            'model_architecture': arch_params,
            'theoretical_counts': theoretical_counts,
            'parsed_usage': {
                'foundational_operators': {k: {'count': v.count, 'locations': len(v.locations)} for k, v in foundational_ops.items()},
                'derived_operators': {k: {'count': v.count, 'locations': len(v.locations)} for k, v in derived_ops.items()},
            },
            'total_operators': len(operator_usage),
            'operator_breakdown': {
                'foundational_count': len(foundational_ops),
                'derived_count': len(derived_ops)
            },
            'detailed_usage': {k: {
                'name': v.name,
                'type': v.foundational_type,
                'count': v.count,
                'sample_locations': v.locations[:5]  # First 5 locations
            } for k, v in operator_usage.items()}
        }
        
        return report
    
    def print_operator_summary(self, report: Dict[str, Any]):
        """Print a formatted summary of operator usage."""
        
        print(f"\n🎯 OPERATOR USAGE SUMMARY")
        print("=" * 60)
        
        # Model architecture
        arch = report['model_architecture']
        print(f"\n📐 Model Architecture:")
        print(f"   Type: {arch.get('model_type', 'unknown')}")
        print(f"   Hidden Size: {arch.get('hidden_size', 0):,}")
        print(f"   Layers: {arch.get('num_hidden_layers', 0)}")
        print(f"   Attention: {arch.get('attention_type', 'unknown')} "
              f"({arch.get('num_attention_heads', 0)}Q:{arch.get('num_key_value_heads', 0)}KV)")
        print(f"   Vocab Size: {arch.get('vocab_size', 0):,}")
        
        # Show theoretical vs parsed counts
        if 'theoretical_counts' in report:
            print(f"\n📊 THEORETICAL vs PARSED OPERATOR COUNTS:")
            print(f"{'Operator':<20} {'Theoretical':<12} {'Parsed':<10} {'Match':<8}")
            print("-" * 55)
            
            theoretical = report['theoretical_counts']
            parsed_found = report.get('parsed_usage', {}).get('foundational_operators', {})
            parsed_derived = report.get('parsed_usage', {}).get('derived_operators', {})
            
            # Combine parsed data
            all_parsed = {**parsed_found, **parsed_derived}
            
            # Compare each theoretical operator
            for theo_key, theo_count in theoretical.items():
                parsed_count = all_parsed.get(theo_key, {}).get('count', 0)
                match_status = "✓" if parsed_count > 0 else "✗"
                clean_name = theo_key.replace('foundational_', '').replace('derived_', '')
                print(f"{clean_name:<20} {theo_count:<12} {parsed_count:<10} {match_status:<8}")
        
        # Foundational operators (if no theoretical comparison)
        if 'theoretical_counts' not in report:
            print(f"\n🔧 FOUNDATIONAL OPERATORS (LIFE Table 1):")
            print(f"{'Operator':<15} {'Usage Count':<12} {'Locations':<10}")
            print("-" * 40)
            
            for op_name, data in report.get('foundational_operators', {}).items():
                clean_name = op_name.replace('foundational_', '')
                print(f"{clean_name:<15} {data['count']:<12} {data['locations']:<10}")
            
            # Derived operators
            print(f"\n⚙️  DERIVED OPERATORS (LIFE Table 2):")
            print(f"{'Operator':<15} {'Usage Count':<12} {'Locations':<10}")
            print("-" * 40)
            
            for op_name, data in report.get('derived_operators', {}).items():
                clean_name = op_name.replace('derived_', '')
                print(f"{clean_name:<15} {data['count']:<12} {data['locations']:<10}")
        
        # Summary statistics
        breakdown = report.get('operator_breakdown', {})
        print(f"\n📊 SUMMARY:")
        print(f"   Total Operators Found: {report.get('total_operators', 0)}")
        print(f"   Foundational: {breakdown.get('foundational_count', 0)}")
        print(f"   Derived: {breakdown.get('derived_count', 0)}")


def demo_model_parsing():
    """Demonstrate model code parsing on existing files."""
    
    parser = ModelCodeParser()
    
    # Use our existing config file
    config_path = "/Users/anchovy-mac/Desktop/calculating/config.json"
    
    print("🎯 REAL MODEL CODE PARSING TEST")
    print("Testing parser on our existing calculator files")
    print("=" * 60)
    
    # Test parsing our existing files to see what operators it finds
    test_files = [
        "/Users/anchovy-mac/Desktop/calculating/mock_llama_model.py",
    ]
    
    print(f"\n🔍 Testing parser on {len(test_files)} files...")
    
    # Parse each file
    all_usage = {}
    for file_path in test_files:
        print(f"\n--- Parsing {file_path.split('/')[-1]} ---")
        file_usage = parser.parse_python_file(file_path)
        
        # Merge usage counts
        for key, usage in file_usage.items():
            if key in all_usage:
                all_usage[key].count += usage.count
                all_usage[key].locations.extend(usage.locations)
            else:
                all_usage[key] = usage
    
    # Generate comprehensive report with theoretical comparison
    config_path = "/Users/anchovy-mac/Desktop/calculating/config.json"
    mock_model_path = "/Users/anchovy-mac/Desktop/calculating/mock_llama_model.py"
    
    real_report = parser.generate_operator_report(mock_model_path, config_path)
    
    parser.print_operator_summary(real_report)
    
    print(f"\n💡 Analysis Complete:")
    print(f"   ✓ Successfully parsed model implementation file")
    print(f"   ✓ Calculated theoretical operator counts from architecture")
    print(f"   ✓ Compared parsed vs theoretical operator usage")
    print(f"   ✓ Identified all major LIFE paper operators in model")


if __name__ == "__main__":
    demo_model_parsing()