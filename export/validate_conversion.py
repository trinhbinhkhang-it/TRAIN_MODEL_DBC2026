"""
Validate model conversion consistency across PyTorch, ONNX, and TFLite
"""

import os
import sys
import argparse
import yaml
import logging
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
    """Load PyTorch model"""
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


def run_pytorch_inference(model, input_tensor, device):
    """Run PyTorch inference"""
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[:, 1]
    return output.cpu().numpy(), prob.cpu().numpy()


def run_onnx_inference(onnx_path, input_np):
    """Run ONNX Runtime inference"""
    if not ONNX_AVAILABLE:
        return None, None
    
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    ort_inputs = {input_name: input_np}
    ort_outputs = session.run([output_name], ort_inputs)
    onnx_output = ort_outputs[0]
    
    # Softmax
    onnx_prob = torch.softmax(torch.from_numpy(onnx_output), dim=1)[:, 1].numpy()
    return onnx_output, onnx_prob


def run_tflite_inference(tflite_path, input_np):
    """Run TFLite inference"""
    if not TF_AVAILABLE:
        return None, None
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Handle quantization
    if input_details[0]['dtype'] == np.uint8:
        scale, zero_point = input_details[0]['quantization']
        input_quantized = (input_np / scale + zero_point).astype(np.uint8)
        interpreter.set_tensor(input_details[0]['index'], input_quantized)
    else:
        interpreter.set_tensor(input_details[0]['index'], input_np.astype(np.float32))
    
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    # Handle quantized output
    if output_details[0]['dtype'] == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        tflite_output = (tflite_output.astype(np.float32) - zero_point) * scale
    
    tflite_prob = torch.softmax(torch.from_numpy(tflite_output), dim=1)[:, 1].numpy()
    return tflite_output, tflite_prob


def compare_outputs(name1, out1, prob1, name2, out2, prob2):
    """Compare two model outputs"""
    if out1 is None or out2 is None:
        return None
    
    abs_diff = np.abs(out1 - out2)
    rel_diff = abs_diff / (np.abs(out1) + 1e-8)
    
    max_abs_diff = np.max(abs_diff)
    max_rel_diff = np.max(rel_diff)
    mean_abs_diff = np.mean(abs_diff)
    
    # Classification agreement
    pred1 = np.argmax(out1, axis=1)
    pred2 = np.argmax(out2, axis=1)
    agreement = np.mean(pred1 == pred2) * 100
    
    # Probability difference
    prob_diff = np.abs(prob1 - prob2)
    max_prob_diff = np.max(prob_diff)
    mean_prob_diff = np.mean(prob_diff)
    
    results = {
        f'{name1}_vs_{name2}': {
            'max_abs_diff': float(max_abs_diff),
            'max_rel_diff': float(max_rel_diff),
            'mean_abs_diff': float(mean_abs_diff),
            'classification_agreement': float(agreement),
            'max_prob_diff': float(max_prob_diff),
            'mean_prob_diff': float(mean_prob_diff),
            'shape_match': out1.shape == out2.shape,
            'shape1': list(out1.shape),
            'shape2': list(out2.shape)
        }
    }
    
    return results


def test_with_sample_images(model_dir, config, device, logger):
    """Test with sample images from test set"""
    from training.train_swapface import FaceSwapVideoDataset, get_transforms
    from torch.utils.data import DataLoader
    
    # Load test dataset
    val_transform = get_transforms(config, is_train=False)
    test_dataset = FaceSwapVideoDataset(config.get('data_root', 'data/'), 'test', config, val_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # Get a few samples
    samples = []
    for batch in test_loader:
        samples.append({
            'image': batch['image'],
            'label': batch['label'].item(),
            'video_name': batch['video_name'][0],
            'frame_name': batch['frame_name'][0]
        })
        if len(samples) >= 10:
            break
    
    return samples


def main():
    parser = argparse.ArgumentParser(description='Validate model conversion consistency')
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
    parser.add_argument('--output', type=str, default='reports/conversion_validation.json',
                        help='Output validation report')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for PyTorch inference')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of test samples')
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load PyTorch model
    logger.info(f"Loading PyTorch model from {args.model}")
    pytorch_model = load_pytorch_model(args.model, config, device)
    
    # Find model files if not provided
    model_dir = Path(args.model).parent
    base_name = Path(args.model).stem.replace('best_', '').replace('_swapface_model', '')
    
    if args.onnx is None:
        onnx_path = model_dir.parent / 'model_files' / f"{base_name}_fp32.onnx"
        if onnx_path.exists():
            args.onnx = str(onnx_path)
    
    if args.tflite_fp32 is None:
        tflite_path = model_dir.parent / 'model_files' / f"{base_name}_fp32.tflite"
        if tflite_path.exists():
            args.tflite_fp32 = str(tflite_path)
    
    if args.tflite_fp16 is None:
        tflite_path = model_dir.parent / 'model_files' / f"{base_name}_fp16.tflite"
        if tflite_path.exists():
            args.tflite_fp16 = str(tflite_path)
    
    if args.tflite_int8 is None:
        tflite_path = model_dir.parent / 'model_files' / f"{base_name}_int8.tflite"
        if tflite_path.exists():
            args.tflite_int8 = str(tflite_path)
    
    logger.info(f"Models to validate:")
    logger.info(f"  PyTorch: {args.model}")
    logger.info(f"  ONNX: {args.onnx}")
    logger.info(f"  TFLite FP32: {args.tflite_fp32}")
    logger.info(f"  TFLite FP16: {args.tflite_fp16}")
    logger.info(f"  TFLite INT8: {args.tflite_int8}")
    
    # Get test samples
    logger.info("Loading test samples...")
    test_samples = test_with_sample_images(model_dir, config, device, logger)
    logger.info(f"Loaded {len(test_samples)} test samples")
    
    all_results = {
        'model_info': {
            'pytorch': args.model,
            'onnx': args.onnx,
            'tflite_fp32': args.tflite_fp32,
            'tflite_fp16': args.tflite_fp16,
            'tflite_int8': args.tflite_int8
        },
        'per_sample': [],
        'aggregate': {}
    }
    
    # Compare across all samples
    comparisons = []
    
    for i, sample in enumerate(test_samples[:args.num_samples]):
        logger.info(f"Testing sample {i+1}/{min(args.num_samples, len(test_samples))}: "
                    f"{sample['video_name']}/{sample['frame_name']} (label={sample['label']})")
        
        input_tensor = sample['image'].unsqueeze(0)
        input_np = input_tensor.numpy()
        
        sample_result = {
            'sample_id': i,
            'video_name': sample['video_name'],
            'frame_name': sample['frame_name'],
            'true_label': sample['label'],
            'comparisons': {}
        }
        
        # PyTorch inference
        pt_out, pt_prob = run_pytorch_inference(pytorch_model, input_tensor, device)
        pt_pred = np.argmax(pt_out, axis=1)[0]
        sample_result['pytorch'] = {
            'prediction': int(pt_pred),
            'fake_score': float(pt_prob[0]),
            'logits': pt_out[0].tolist()
        }
        
        # ONNX inference
        if args.onnx and ONNX_AVAILABLE:
            onnx_out, onnx_prob = run_onnx_inference(args.onnx, input_np)
            if onnx_out is not None:
                onnx_pred = np.argmax(onnx_out, axis=1)[0]
                sample_result['onnx'] = {
                    'prediction': int(onnx_pred),
                    'fake_score': float(onnx_prob[0]),
                    'logits': onnx_out[0].tolist()
                }
                
                # Compare PyTorch vs ONNX
                comp = compare_outputs('pytorch', pt_out, pt_prob, 'onnx', onnx_out, onnx_prob)
                sample_result['comparisons'].update(comp)
        
        # TFLite FP32
        if args.tflite_fp32 and TF_AVAILABLE:
            tflite_out, tflite_prob = run_tflite_inference(args.tflite_fp32, input_np)
            if tflite_out is not None:
                tflite_pred = np.argmax(tflite_out, axis=1)[0]
                sample_result['tflite_fp32'] = {
                    'prediction': int(tflite_pred),
                    'fake_score': float(tflite_prob[0]),
                    'logits': tflite_out[0].tolist()
                }
                
                comp = compare_outputs('pytorch', pt_out, pt_prob, 'tflite_fp32', tflite_out, tflite_prob)
                sample_result['comparisons'].update(comp)
        
        # TFLite FP16
        if args.tflite_fp16 and TF_AVAILABLE:
            tflite_out, tflite_prob = run_tflite_inference(args.tflite_fp16, input_np)
            if tflite_out is not None:
                tflite_pred = np.argmax(tflite_out, axis=1)[0]
                sample_result['tflite_fp16'] = {
                    'prediction': int(tflite_pred),
                    'fake_score': float(tflite_prob[0]),
                    'logits': tflite_out[0].tolist()
                }
                
                comp = compare_outputs('pytorch', pt_out, pt_prob, 'tflite_fp16', tflite_out, tflite_prob)
                sample_result['comparisons'].update(comp)
        
        # TFLite INT8
        if args.tflite_int8 and TF_AVAILABLE:
            tflite_out, tflite_prob = run_tflite_inference(args.tflite_int8, input_np)
            if tflite_out is not None:
                tflite_pred = np.argmax(tflite_out, axis=1)[0]
                sample_result['tflite_int8'] = {
                    'prediction': int(tflite_pred),
                    'fake_score': float(tflite_prob[0]),
                    'logits': tflite_out[0].tolist()
                }
                
                comp = compare_outputs('pytorch', pt_out, pt_prob, 'tflite_int8', tflite_out, tflite_prob)
                sample_result['comparisons'].update(comp)
        
        all_results['per_sample'].append(sample_result)
        
        # Log sample result
        logger.info(f"  PyTorch: pred={pt_pred}, fake_score={pt_prob[0]:.4f}")
        if 'onnx' in sample_result:
            logger.info(f"  ONNX:    pred={sample_result['onnx']['prediction']}, "
                        f"fake_score={sample_result['onnx']['fake_score']:.4f}")
        if 'tflite_fp32' in sample_result:
            logger.info(f"  TFLite FP32: pred={sample_result['tflite_fp32']['prediction']}, "
                        f"fake_score={sample_result['tflite_fp32']['fake_score']:.4f}")
    
    # Aggregate statistics
    logger.info("\n" + "="*60)
    logger.info("AGGREGATE COMPARISON RESULTS")
    logger.info("="*60)
    
    for comp_key in ['pytorch_vs_onnx', 'pytorch_vs_tflite_fp32', 'pytorch_vs_tflite_fp16', 'pytorch_vs_tflite_int8']:
        agreements = []
        max_prob_diffs = []
        mean_prob_diffs = []
        max_abs_diffs = []
        
        for sample in all_results['per_sample']:
            if comp_key in sample['comparisons']:
                comp = sample['comparisons'][comp_key]
                agreements.append(comp['classification_agreement'])
                max_prob_diffs.append(comp['max_prob_diff'])
                mean_prob_diffs.append(comp['mean_prob_diff'])
                max_abs_diffs.append(comp['max_abs_diff'])
        
        if agreements:
            agg = {
                'mean_agreement': float(np.mean(agreements)),
                'min_agreement': float(np.min(agreements)),
                'max_agreement': float(np.max(agreements)),
                'mean_max_prob_diff': float(np.mean(max_prob_diffs)),
                'max_max_prob_diff': float(np.max(max_prob_diffs)),
                'mean_abs_diff': float(np.mean(max_abs_diffs))
            }
            all_results['aggregate'][comp_key] = agg
            
            logger.info(f"\n{comp_key}:")
            logger.info(f"  Mean Agreement: {agg['mean_agreement']:.2f}% (min: {agg['min_agreement']:.2f}%, max: {agg['max_agreement']:.2f}%)")
            logger.info(f"  Mean Max Prob Diff: {agg['mean_max_prob_diff']:.6f}")
            logger.info(f"  Max Max Prob Diff: {agg['max_max_prob_diff']:.6f}")
            logger.info(f"  Mean Abs Diff: {agg['mean_abs_diff']:.6f}")
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nValidation report saved to {output_path}")
    
    # Print summary table
    logger.info(f"\n{'='*90}")
    logger.info(f"SUMMARY TABLE")
    logger.info(f"{'='*90}")
    logger.info(f"{'Comparison':<30} {'Agreement':<12} {'Max Prob Diff':<15} {'Mean Abs Diff':<15}")
    logger.info(f"{'-'*90}")
    
    for comp_key, agg in all_results['aggregate'].items():
        logger.info(f"{comp_key:<30} {agg['mean_agreement']:<12.2f} {agg['max_max_prob_diff']:<15.6f} {agg['mean_abs_diff']:<15.6f}")
    
    # Overall verdict
    logger.info(f"\n{'='*60}")
    all_passed = True
    for comp_key, agg in all_results['aggregate'].items():
        if agg['mean_agreement'] < 99.0 or agg['max_max_prob_diff'] > 0.01:
            all_passed = False
            logger.warning(f"⚠ {comp_key}: Agreement={agg['mean_agreement']:.2f}%, MaxProbDiff={agg['max_max_prob_diff']:.6f}")
        else:
            logger.info(f"✓ {comp_key}: Agreement={agg['mean_agreement']:.2f}%, MaxProbDiff={agg['max_max_prob_diff']:.6f}")
    
    if all_passed:
        logger.info("\n✓ ALL VALIDATIONS PASSED - Models are consistent!")
    else:
        logger.warning("\n⚠ SOME VALIDATIONS FAILED - Review differences carefully")
    
    logger.info("Validation completed!")


if __name__ == "__main__":
    main()