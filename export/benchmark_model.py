"""
Benchmark model performance across PyTorch, ONNX, and TFLite
Measures latency, model size, and accuracy
"""

import os
import sys
import argparse
import yaml
import logging
import time
import json
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_pytorch_model(model_path, config, device):
    from training.swapface_detector import create_swapface_detector
    
    model_config = config.get('model', {})
    model = create_swapface_detector(model_config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def get_model_size_mb(model_path):
    """Get model file size in MB"""
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def benchmark_pytorch(model, input_shape, device, num_runs=100, warmup=10):
    """Benchmark PyTorch model"""
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Synchronize for accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    return {
        'mean_latency_ms': float(np.mean(times)),
        'std_latency_ms': float(np.std(times)),
        'p50_latency_ms': float(np.percentile(times, 50)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'p99_latency_ms': float(np.percentile(times, 99)),
        'min_latency_ms': float(np.min(times)),
        'max_latency_ms': float(np.max(times)),
        'fps': float(1000 / np.mean(times))
    }


def benchmark_onnx(onnx_path, input_shape, num_runs=100, warmup=10, provider='CPUExecutionProvider'):
    """Benchmark ONNX Runtime"""
    if not ONNX_AVAILABLE:
        return None
    
    session = ort.InferenceSession(onnx_path, providers=[provider])
    input_name = session.get_inputs()[0].name
    
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    ort_inputs = {input_name: dummy_input}
    
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, ort_inputs)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, ort_inputs)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return {
        'mean_latency_ms': float(np.mean(times)),
        'std_latency_ms': float(np.std(times)),
        'p50_latency_ms': float(np.percentile(times, 50)),
        'p95_latency_ms': float(np.percentile(times, 95)),
        'p99_latency_ms': float(np.percentile(times, 99)),
        'min_latency_ms': float(np.min(times)),
        'max_latency_ms': float(np.max(times)),
        'fps': float(1000 / np.mean(times))
    }


def benchmark_tflite(tflite_path, input_shape, num_runs=100, warmup=10, use_delegate=None):
    """Benchmark TFLite model"""
    if not TF_AVAILABLE:
        return None
    
    # Setup delegates
    delegates = []
    if use_delegate == 'gpu':
        try:
            from tensorflow.lite.python.interpreter import load_delegate
            delegates.append(load_delegate('libtensorflowlite_gpu_delegate.so'))
        except:
            logging.warning("GPU delegate not available")
    elif use_delegate == 'nnap':
        try:
            from tensorflow.lite.python.interpreter import load_delegate
            delegates.append(load_delegate('libnnapi_delegate.so'))
        except:
            logging.warning("NNAPI delegate not available")
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
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
        'max_latency_ms': float(np.max(times)),
        'fps': float(1000 / np.mean(times)),
        'delegate': use_delegate or 'cpu'
    }


def run_accuracy_evaluation(model_path, config, device, model_type='pytorch'):
    """Run accuracy evaluation on test set"""
    from training.evaluate_swapface import evaluate_model
    from training.train_swapface import FaceSwapVideoDataset, get_transforms
    from torch.utils.data import DataLoader
    
    if model_type == 'pytorch':
        model = load_pytorch_model(model_path, config, device)
        val_transform = get_transforms(config, is_train=False)
        test_dataset = FaceSwapVideoDataset(config.get('data_root', 'data/'), 'test', config, val_transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        results = evaluate_model(model, test_loader, device, logging.getLogger(__name__), 'test')
        return results
    
    # For ONNX/TFLite, would need custom evaluation
    return None


def main():
    parser = argparse.ArgumentParser(description='Benchmark model performance')
    parser.add_argument('--config', type=str, default='configs/swapface_detector.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to PyTorch model checkpoint')
    parser.add_argument('--onnx', type=str, default=None,
                        help='Path to ONNX model')
    parser.add_argument('--tflite_fp32', type=str, default=None,
                        help='Path to TFLite FP32 model')
    parser.add_argument('--tflite_fp16', type=str, default=None,
                        help='Path to TFLite FP16 model')
    parser.add_argument('--tflite_int8', type=str, default=None,
                        help='Path to TFLite INT8 model')
    parser.add_argument('--output', type=str, default='reports/benchmark_report.json',
                        help='Output benchmark report')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for PyTorch inference')
    parser.add_argument('--input_size', type=int, nargs=2, default=[224, 224],
                        help='Input size H W')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--num_runs', type=int, default=100,
                        help='Number of benchmark runs')
    parser.add_argument('--onnx_provider', type=str, default='CPUExecutionProvider',
                        choices=['CPUExecutionProvider', 'CUDAExecutionProvider'],
                        help='ONNX Runtime provider')
    parser.add_argument('--tflite_delegate', type=str, default='cpu',
                        choices=['cpu', 'gpu', 'nnap'],
                        help='TFLite delegate')
    parser.add_argument('--evaluate_accuracy', action='store_true',
                        help='Run accuracy evaluation on test set')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Input shape
    input_shape = (args.batch_size, 3, args.input_size[0], args.input_size[1])
    logger.info(f"Input shape: {input_shape}")
    
    # Find model files
    model_dir = Path(args.model).parent
    base_name = Path(args.model).stem.replace('best_', '').replace('_swapface_model', '')
    
    model_files = {
        'pytorch': args.model,
        'onnx': args.onnx,
        'tflite_fp32': args.tflite_fp32,
        'tflite_fp16': args.tflite_fp16,
        'tflite_int8': args.tflite_int8
    }
    
    # Auto-detect if not provided
    for key in ['onnx', 'tflite_fp32', 'tflite_fp16', 'tflite_int8']:
        if model_files[key] is None:
            ext = '.onnx' if key == 'onnx' else '.tflite'
            quant = '' if key in ['onnx', 'tflite_fp32'] else f'_{key.split("_")[-1]}'
            guess_path = model_dir.parent / 'model_files' / f"{base_name}{quant}{ext}"
            if guess_path.exists():
                model_files[key] = str(guess_path)
    
    logger.info("Models to benchmark:")
    for key, path in model_files.items():
        if path:
            size = get_model_size_mb(path) if os.path.exists(path) else 0
            logger.info(f"  {key}: {path} ({size:.2f} MB)")
    
    # Load PyTorch model for benchmarking
    pytorch_model = load_pytorch_model(args.model, config, device)
    
    # Benchmark all models
    results = {
        'config': {
            'input_shape': list(input_shape),
            'num_runs': args.num_runs,
            'device': str(device),
            'onnx_provider': args.onnx_provider,
            'tflite_delegate': args.tflite_delegate
        },
        'models': {}
    }
    
    # PyTorch benchmark
    logger.info("\n" + "="*60)
    logger.info("Benchmarking PyTorch")
    logger.info("="*60)
    
    pt_bench = benchmark_pytorch(pytorch_model, input_shape, device, args.num_runs)
    pt_size = get_model_size_mb(args.model)
    
    results['models']['pytorch'] = {
        'benchmark': pt_bench,
        'size_mb': pt_size,
        'framework': 'PyTorch',
        'precision': 'FP32'
    }
    
    logger.info(f"  Size: {pt_size:.2f} MB")
    logger.info(f"  Mean Latency: {pt_bench['mean_latency_ms']:.2f} ms")
    logger.info(f"  P95 Latency: {pt_bench['p95_latency_ms']:.2f} ms")
    logger.info(f"  FPS: {pt_bench['fps']:.2f}")
    
    # ONNX benchmark
    if model_files['onnx'] and ONNX_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info(f"Benchmarking ONNX ({args.onnx_provider})")
        logger.info("="*60)
        
        onnx_bench = benchmark_onnx(model_files['onnx'], input_shape, args.num_runs, 
                                    provider=args.onnx_provider)
        onnx_size = get_model_size_mb(model_files['onnx'])
        
        results['models']['onnx'] = {
            'benchmark': onnx_bench,
            'size_mb': onnx_size,
            'framework': 'ONNX Runtime',
            'provider': args.onnx_provider
        }
        
        logger.info(f"  Size: {onnx_size:.2f} MB")
        logger.info(f"  Mean Latency: {onnx_bench['mean_latency_ms']:.2f} ms")
        logger.info(f"  P95 Latency: {onnx_bench['p95_latency_ms']:.2f} ms")
        logger.info(f"  FPS: {onnx_bench['fps']:.2f}")
    
    # TFLite benchmarks
    tflite_models = {
        'tflite_fp32': 'FP32',
        'tflite_fp16': 'FP16',
        'tflite_int8': 'INT8'
    }
    
    for key, precision in tflite_models.items():
        if model_files[key] and TF_AVAILABLE:
            logger.info("\n" + "="*60)
            logger.info(f"Benchmarking TFLite {precision} ({args.tflite_delegate})")
            logger.info("="*60)
            
            tflite_bench = benchmark_tflite(
                model_files[key], input_shape, args.num_runs, 
                use_delegate=args.tflite_delegate if args.tflite_delegate != 'cpu' else None)
            tflite_size = get_model_size_mb(model_files[key])
            
            results['models'][key] = {
                'benchmark': tflite_bench,
                'size_mb': tflite_size,
                'framework': 'TensorFlow Lite',
                'precision': precision,
                'delegate': tflite_bench.get('delegate', 'cpu')
            }
            
            logger.info(f"  Size: {tflite_size:.2f} MB")
            logger.info(f"  Mean Latency: {tflite_bench['mean_latency_ms']:.2f} ms")
            logger.info(f"  P95 Latency: {tflite_bench['p95_latency_ms']:.2f} ms")
            logger.info(f"  FPS: {tflite_bench['fps']:.2f}")
    
    # Accuracy evaluation
    if args.evaluate_accuracy:
        logger.info("\n" + "="*60)
        logger.info("Running Accuracy Evaluation")
        logger.info("="*60)
        
        acc_results = run_accuracy_evaluation(args.model, config, device, 'pytorch')
        if acc_results:
            results['accuracy'] = acc_results
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nBenchmark report saved to {output_path}")
    
    # Print summary table
    logger.info(f"\n{'='*100}")
    logger.info(f"BENCHMARK SUMMARY")
    logger.info(f"{'='*100}")
    logger.info(f"{'Model':<25} {'Framework':<20} {'Precision':<10} {'Size (MB)':<12} {'Latency (ms)':<15} {'P95 (ms)':<12} {'FPS':<8}")
    logger.info(f"{'-'*100}")
    
    for key, model_result in results['models'].items():
        bench = model_result['benchmark']
        framework = model_result.get('framework', '')
        precision = model_result.get('precision', '')
        delegate = model_result.get('delegate', '')
        provider = model_result.get('provider', '')
        
        fw_str = framework
        if delegate:
            fw_str += f" ({delegate})"
        if provider:
            fw_str += f" ({provider})"
        
        logger.info(f"{key:<25} {fw_str:<20} {precision:<10} {model_result['size_mb']:<12.2f} "
                    f"{bench['mean_latency_ms']:<15.2f} {bench['p95_latency_ms']:<12.2f} {bench['fps']:<8.2f}")
    
    logger.info("Benchmark completed!")


if __name__ == "__main__":
    main()