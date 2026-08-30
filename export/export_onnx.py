"""
Export PyTorch model to ONNX format
Validates ONNX output against PyTorch
"""

import os
import sys
import argparse
import yaml
import logging
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.swapface_detector import create_swapface_detector
from training.efficientnet_b0_backbone import create_efficientnet_b0


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(model_path, config, device):
    """Load PyTorch model from checkpoint"""
    model_config = config.get('model', {})
    model = create_swapface_detector(model_config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def export_to_onnx(model, input_shape, onnx_path, opset_version=17, 
                   input_name='input', output_name='output', dynamic_batch=False):
    """Export model to ONNX format"""
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Dynamic axes for batch size
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes[input_name] = {0: 'batch_size'}
        dynamic_axes[output_name] = {0: 'batch_size'}
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    return onnx_model


def simplify_onnx(onnx_path, simplified_path):
    """Simplify ONNX model using onnxsim"""
    try:
        import onnxsim
        model = onnx.load(onnx_path)
        model_simp, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_simp, simplified_path)
            return True
        else:
            print("ONNX simplification check failed")
            return False
    except ImportError:
        print("onnxsim not available, skipping simplification")
        return False


def validate_onnx(onnx_path, pytorch_model, input_shape, device, 
                  input_name='input', output_name='output', rtol=1e-3, atol=1e-5):
    """Validate ONNX output matches PyTorch output"""
    
    # Create test input
    test_input = torch.randn(*input_shape).to(device)
    test_input_np = test_input.cpu().numpy()
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    
    pytorch_output_np = pytorch_output.cpu().numpy()
    
    # ONNX Runtime inference
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {input_name: test_input_np}
    ort_outputs = ort_session.run([output_name], ort_inputs)
    onnx_output_np = ort_outputs[0]
    
    # Compare
    abs_diff = np.abs(pytorch_output_np - onnx_output_np)
    rel_diff = abs_diff / (np.abs(pytorch_output_np) + 1e-8)
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Classification agreement
    pytorch_pred = np.argmax(pytorch_output_np, axis=1)
    onnx_pred = np.argmax(onnx_output_np, axis=1)
    agreement = np.mean(pytorch_pred == onnx_pred) * 100
    
    # Softmax probability difference (for fake class)
    pytorch_prob = torch.softmax(torch.from_numpy(pytorch_output_np), dim=1)[:, 1].numpy()
    onnx_prob = torch.softmax(torch.from_numpy(onnx_output_np), dim=1)[:, 1].numpy()
    prob_diff = np.abs(pytorch_prob - onnx_prob)
    max_prob_diff = np.max(prob_diff)
    mean_prob_diff = np.mean(prob_diff)
    
    results = {
        'max_abs_diff': float(max_abs_diff),
        'max_rel_diff': float(max_rel_diff),
        'mean_abs_diff': float(mean_abs_diff),
        'classification_agreement': float(agreement),
        'max_prob_diff': float(max_prob_diff),
        'mean_prob_diff': float(mean_prob_diff),
        'pytorch_output_shape': list(pytorch_output_np.shape),
        'onnx_output_shape': list(onnx_output_np.shape)
    }
    
    return results


def benchmark_onnx(onnx_path, input_shape, num_runs=100, warmup=10):
    """Benchmark ONNX Runtime inference"""
    import time
    
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {input_name: dummy_input}
    
    # Warmup
    for _ in range(warmup):
        _ = ort_session.run(None, ort_inputs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    return {
        'mean_latency_ms': float(np.mean(times)),
        'std_latency_ms': float(np.std(times)),
        'p50_latency_ms': float(np.percentile(times, 50)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'p99_latency_ms': float(np.percentile(times, 99)),
        'min_latency_ms': float(np.min(times)),
        'max_latency_ms': float(np.max(times))
    }


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--config', type=str, default='configs/swapface_detector.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch model checkpoint')
    parser.add_argument('--output', type=str, default='model_files/swapface_detector_fp32.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for export (1 for mobile)')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                        help='Input size H W')
    parser.add_argument('--dynamic_batch', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model with onnxsim')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate ONNX vs PyTorch output')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark ONNX Runtime')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for PyTorch inference')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model, config, device)
    
    # Input shape
    input_shape = (args.batch_size, 3, args.input_size[0], args.input_size[1])
    logger.info(f"Input shape: {input_shape}")
    
    # Export
    logger.info(f"Exporting to ONNX: {args.output}")
    onnx_model = export_to_onnx(
        model, input_shape, args.output,
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch
    )
    logger.info("ONNX export successful!")
    
    # Simplify
    if args.simplify:
        simplified_path = args.output.replace('.onnx', '_simplified.onnx')
        logger.info(f"Simplifying ONNX model to {simplified_path}")
        if simplify_onnx(args.output, simplified_path):
            logger.info("ONNX simplification successful!")
            args.output = simplified_path  # Use simplified for validation
        else:
            logger.warning("ONNX simplification failed, using original")
    
    # Validate
    if args.validate:
        logger.info("Validating ONNX vs PyTorch...")
        val_results = validate_onnx(args.output, model, input_shape, device)
        
        logger.info(f"Validation Results:")
        logger.info(f"  Max Absolute Difference: {val_results['max_abs_diff']:.6f}")
        logger.info(f"  Max Relative Difference: {val_results['max_rel_diff']:.6f}")
        logger.info(f"  Mean Absolute Difference: {val_results['mean_abs_diff']:.6f}")
        logger.info(f"  Classification Agreement: {val_results['classification_agreement']:.2f}%")
        logger.info(f"  Max Probability Difference: {val_results['max_prob_diff']:.6f}")
        logger.info(f"  Mean Probability Difference: {val_results['mean_prob_diff']:.6f}")
        
        # Check tolerance
        if val_results['max_abs_diff'] < 1e-3 and val_results['classification_agreement'] > 99.9:
            logger.info("✓ ONNX validation PASSED")
        else:
            logger.warning("⚠ ONNX validation shows differences - review carefully")
        
        # Save validation report
        import json
        report_path = Path(args.output).with_suffix('.validation.json')
        with open(report_path, 'w') as f:
            json.dump(val_results, f, indent=2)
        logger.info(f"Validation report saved to {report_path}")
    
    # Benchmark
    if args.benchmark:
        logger.info("Benchmarking ONNX Runtime...")
        bench_results = benchmark_onnx(args.output, input_shape)
        
        logger.info(f"Benchmark Results (CPU):")
        logger.info(f"  Mean Latency: {bench_results['mean_latency_ms']:.2f} ms")
        logger.info(f"  P50 Latency:  {bench_results['p50_latency_ms']:.2f} ms")
        logger.info(f"  P95 Latency:  {bench_results['p95_latency_ms']:.2f} ms")
        logger.info(f"  P99 Latency:  {bench_results['p99_latency_ms']:.2f} ms")
        logger.info(f"  Std Dev:      {bench_results['std_latency_ms']:.2f} ms")
        
        # Save benchmark report
        import json
        bench_path = Path(args.output).with_suffix('.benchmark.json')
        with open(bench_path, 'w') as f:
            json.dump(bench_results, f, indent=2)
        logger.info(f"Benchmark report saved to {bench_path}")
    
    # Print model info
    logger.info(f"\nModel Info:")
    logger.info(f"  Input: {input_name} {list(input_shape)}")
    logger.info(f"  Output: {output_name} [batch, 2]")
    logger.info(f"  Opset: {args.opset}")
    logger.info(f"  Dynamic Batch: {args.dynamic_batch}")
    
    logger.info("Export completed!")


if __name__ == "__main__":
    main()