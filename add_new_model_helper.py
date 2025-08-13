"""
Helper script to add new models to the benchmark
Usage: python add_new_model_helper.py --config_path /path/to/new_model_config.json
"""

import json
import argparse
import os

def extract_model_specs_from_config(config_path):
    """Extract model specifications from HuggingFace config.json"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract key parameters
    model_specs = {
        'hidden_size': config.get('hidden_size'),
        'intermediate_size': config.get('intermediate_size'),
        'num_attention_heads': config.get('num_attention_heads'),
        'num_key_value_heads': config.get('num_key_value_heads', config.get('num_attention_heads')),  # Fallback for MHA
        'head_dim': config.get('hidden_size', 0) // config.get('num_attention_heads', 1),
        'layers': config.get('num_hidden_layers'),
        'vocab_size': config.get('vocab_size'),
        'model_type': config.get('model_type', 'unknown'),
        'architectures': config.get('architectures', [])
    }
    
    return model_specs

def update_benchmark_with_new_model(model_name, model_specs):
    """Update operator_efficiency_benchmark.py with new model"""
    
    benchmark_file = 'operator_efficiency_benchmark.py'
    
    if not os.path.exists(benchmark_file):
        print(f"❌ {benchmark_file} not found!")
        return
    
    # Read current file
    with open(benchmark_file, 'r') as f:
        content = f.read()
    
    # Find the llama_models dictionary
    models_start = content.find("llama_models = {")
    if models_start == -1:
        print("❌ Could not find llama_models dictionary!")
        return
    
    # Find the end of the dictionary
    models_end = content.find("}", models_start)
    if models_end == -1:
        print("❌ Could not find end of llama_models dictionary!")
        return
    
    # Generate new model entry
    new_model_entry = f"""
            '{model_name}': {{
                'hidden_size': {model_specs['hidden_size']},
                'intermediate_size': {model_specs['intermediate_size']},
                'num_attention_heads': {model_specs['num_attention_heads']},
                'num_key_value_heads': {model_specs['num_key_value_heads']},
                'head_dim': {model_specs['head_dim']},
                'layers': {model_specs['layers']},
                'vocab_size': {model_specs['vocab_size']}
            }},"""
    
    # Insert new model before the closing brace
    new_content = content[:models_end] + new_model_entry + "\n        " + content[models_end:]
    
    # Write back to file
    with open(benchmark_file, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Added {model_name} to benchmark!")
    print(f"📊 Model specs: {model_specs}")

def main():
    parser = argparse.ArgumentParser(description='Add new model to benchmark')
    parser.add_argument('--config_path', required=True, help='Path to model config.json')
    parser.add_argument('--model_name', required=True, help='Name for the model (e.g., Mistral_7B)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"❌ Config file not found: {args.config_path}")
        return
    
    # Extract model specifications
    model_specs = extract_model_specs_from_config(args.config_path)
    
    print(f"🔍 Extracted model specs from {args.config_path}:")
    for key, value in model_specs.items():
        print(f"   {key}: {value}")
    
    # Update benchmark file
    update_benchmark_with_new_model(args.model_name, model_specs)
    
    print(f"\n🚀 Ready to benchmark {args.model_name}!")
    print(f"   Run: python operator_efficiency_benchmark.py")

if __name__ == "__main__":
    main()