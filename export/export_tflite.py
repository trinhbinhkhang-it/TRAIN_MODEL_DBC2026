"""
Convert ONNX model to TFLite (FP32, FP16, INT8)
Validates TFLite output against PyTorch/ONNX
"""

import os
import sys
import argparse
import yaml
import logging
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. TFLite conversion will not work.")

try:
    import tf2onnx
    TF2ONNX_AVAILABLE = True
except ImportError:
    TF2ONNX_AVAILABLE = False
    print("Warning: tf2onnx not available.")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def convert_onnx_to_tflite(onnx_path, tflite_path, input_shape, 
                           quantization='fp32', representative_dataset=None):
    """
    Convert ONNX to TFLite with optional quantization
    quantization: 'fp32', 'fp16', 'int8'
    """
    if not TF_AVAILABLE or not TF2ONNX_AVAILABLE:
        raise RuntimeError("TensorFlow and tf2onnx are required for TFLite conversion")
    
    import tf2onnx
    import onnx
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert ONNX to TensorFlow SavedModel
    # Using tf2onnx for ONNX -> TF conversion
    tf_model_path = tflite_path.replace('.tflite', '_tf_savedmodel')
    
    # Convert ONNX to TF
    tf_model, _ = tf2onnx.convert.from_onnx(
        onnx_model,
        opset=17,
        output_path=tf_model_path
    )
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    
    # Set input shape
    converter.input_shapes = {'input': input_shape}
    
    if quantization == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        
        if representative_dataset is not None:
            def rep_data_gen():
                for data in representative_dataset:
                    yield [data.astype(np.float32)]
            converter.representative_dataset = rep_data_gen
        else:
            logging.warning("No representative dataset provided for INT8 quantization")
    # FP32 is default
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    logging.info(f"TFLite model saved to {tflite_path}")
    
    # Get model size
    model_size = len(tflite_model) / (1024 * 1024)  # MB
    logging.info(f"Model size: {model_size:.2f} MB")
    
    return tflite_model


def create_representative_dataset(data_loader, num_samples=100):
    """Create representative dataset for INT8 quantization"""
    data = []
    for i, batch in enumerate(data_loader):
        if i >= num_samples:
            break
        img = batch['image'].numpy()
        data.append(img)
    return np.concatenate(data, axis=0) if data else None


def validate_tflite(tflite_path, pytorch_model, input_shape, device, 
                    input_name='input', output_name='output'):
    """Validate TFLite output matches PyTorch"""
    
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for TFLite validation")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logging.info(f"TFLite Input: {input_details}")
    logging.info(f"TFLite Output: {output_details}")
    
    # Create test input
    test_input = torch.randn(*input_shape).to(device)
    test_input_np = test_input.cpu().numpy()
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    pytorch_output_np = pytorch_output.cpu().numpy()
    
    # TFLite inference
    # Handle quantization
    if input_details[0]['dtype'] == np.uint8:
        # Quantized input
        scale, zero_point = input_details[0]['quantization']
        test_input_quantized = (test_input_np / scale + zero_point).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], test_input_quantized)
    else:
        interpreter.set_tensor(input_details[0]['index'], test_input_np.astype(np.float32))
    
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle quantized output
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        tflite_output = (tflite_output.astype(np.float32) - zero_point) * scale
    
    # Compare
    abs_diff = np.abs(pytorch_output_np - tflite_output)
    rel_diff = abs_diff / (np.abs(pytorch_output_np) + 1e-8)
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Classification agreement
    pytorch_pred = np.argmax(pytorch_output_np, axis=1)
    tflite_pred = np.argmax(tflite_output, axis=1)
    agreement = np.mean(pytorch_pred == tflite_pred) * 100
    
    # Probability difference
    pytorch_prob = torch.softmax(torch.from_numpy(pytorch_output_np), dim=1)[:, 1].numpy()
    tflite_prob = torch.softmax(torch.from_numpy(tflite_output), dim=1)[:, 1].numpy()
    prob_diff = np.abs(pytorch_prob - tflite_prob)
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
        'tflite_output_shape': list(tflite_output.shape),
        'input_dtype': str(input_details[0]['dtype']),
        'output_dtype': str(output_details[0]['dtype'])
    }
    
    return results


def benchmark_tflite(tflite_path, input_shape, num_runs=100, warmup=10):
    """Benchmark TFLite inference"""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow required for TFLite benchmark")
    
    import time
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Handle quantization
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        dummy_input = (dummy_input / scale + zero_point).astype(np.uint8)
    
    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        'mean_latency_ms': float(np.mean(times)),
        'std_latency_ms': float(np.std(times)),
        'p50_latency_ms': float(np.percentile(times, 50)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'p99_latency_ms': float(np.percentile(times, 99)),
        'min_latency_ms': float(np.min(times)),
        'max_latency_ms': float(np.max(times))
    }


def get_model_size_mb(model_path):
    """Get model size in MB"""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TFLite')
    parser.add_argument('--config', type=str, default='configs/swapface_detector.yaml',
                        help='Path to config file')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--output_dir', type=str, default='model_files/',
                        help='Output directory for TFLite models')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                        help='Input size H W')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--quantize', type=str, nargs='+', 
                        default=['fp32', 'fp16', 'int8'],
                        choices=['fp32', 'fp16', 'int8'],
                        help='Quantization modes to generate')
    parser.add_argument('--representative_data', type=str, default=None,
                        help='Path to representative dataset for INT8')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate TFLite vs PyTorch')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark TFLite inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for PyTorch inference')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    if not TF_AVAILABLE or not TF2ONNX_AVAILABLE:
        logger.error("TensorFlow and tf2onnx are required. Please install them.")
        return
    
    # Load config
    config = load_config(args.config)
    
    # Load PyTorch model for validation
    sys.path.append(str(Path(__file__).parent.parent))
    from training.swapface_detector import create_swapface_detector
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_config = config.get('model', {})
    pytorch_model = create_swapface_detector(model_config).to(device)
    
    # Find checkpoint
    checkpoint_path = config.get('paths', {}).get('checkpoints', 'checkpoints/')
    best_model = Path(checkpoint_path) / 'best_swapface_model.pth'
    if best_model.exists():
        checkpoint = torch.load(best_model, map_location=device)
        if 'model_state_dict' in checkpoint:
            pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            pytorch_model.load_state_dict(checkpoint)
        logger.info(f"Loaded PyTorch model for validation from {best_model}")
    else:
        logger.warning("No PyTorch checkpoint found for validation")
    
    pytorch_model.eval()
    
    # Input shape
    input_shape = [args.batch_size, 3, args.input_size[0], args.input_size[1]]
    logger.info(f"Input shape: {input_shape}")
    
    # Create representative dataset for INT8
    representative_dataset = None
    if 'int8' in args.quantize:
        if args.representative_data:
            # Load from file
            representative_dataset = np.load(args.representative_data)
        else:
            logger.warning("No representative dataset for INT8, using random data")
            representative_dataset = np.random.randn(100, *input_shape[1:]).astype(np.float32)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(args.onnx).stem
    
    all_results = {}
    
    for quant_mode in args.quantize:
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting to TFLite {quant_mode.upper()}")
        logger.info(f"{'='*60}")
        
        tflite_path = output_dir / f"{base_name}_{quant_mode}.tflite"
        
        try:
            # Convert
            convert_onnx_to_tflite(
                args.onnx, str(tflite_path), input_shape,
                quantization=quant_mode,
                representative_dataset=representative_dataset
            )
            
            # Model size
            size_mb = get_model_size_mb(tflite_path)
            logger.info(f"Model size: {size_mb:.2f} MB")
            
            # Validate
            if args.validate:
                logger.info(f"Validating TFLite {quant_mode} vs PyTorch...")
                val_results = validate_tflite(
                    str(tflite_path), pytorch_model, input_shape, device)
                
                logger.info(f"  Max Abs Diff: {val_results['max_abs_diff']:.6f}")
                logger.info(f"  Classification Agreement: {val_results['classification_agreement']:.2f}%")
                logger.info(f"  Max Prob Diff: {val_results['max_prob_diff']:.6f}")
                
                all_results[quant_mode] = {
                    'validation': val_results,
                    'size_mb': size_mb
                }
            
            # Benchmark
            if args.benchmark:
                logger.info(f"Benchmarking TFLite {quant_mode}...")
                bench_results = benchmark_tflite(str(tflite_path), input_shape)
                
                logger.info(f"  Mean Latency: {bench_results['mean_latency_ms']:.2f} ms")
                logger.info(f"  P95 Latency:  {bench_results['p95_latency_ms']:.2f} ms")
                
                if quant_mode in all_results:
                    all_results[quant_mode]['benchmark'] = bench_results
                else:
                    all_results[quant_mode] = {'benchmark': bench_results, 'size_mb': size_mb}
            
        except Exception as e:
            logger.error(f"Failed to convert {quant_mode}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save summary report
    import json
    report_path = output_dir / f"{base_name}_conversion_report.json"
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nConversion report saved to {report_path}")
    
    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info(f"CONVERSION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"{'Model':<30} {'Size (MB)':<12} {'Agreement':<12} {'Max Prob Diff':<15} {'Latency (ms)':<12}")
    logger.info(f"{'-'*80}")
    
    for quant_mode, results in all_results.items():
        size = results.get('size_mb', 0)
        agreement = results.get('validation', {}).get('classification_agreement', 0)
        prob_diff = results.get('validation', {}).get('max_prob_diff', 0)
        latency = results.get('benchmark', {}).get('mean_latency_ms', 0)
        logger.info(f"{quant_mode:<30} {size:<12.2f} {agreement:<12.2f} {prob_diff:<15.6f} {latency:<12.2f}")
    
    logger.info("TFLite conversion completed!")


if __name__ == "__main__":
    main()