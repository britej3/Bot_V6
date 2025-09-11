"""
Model Optimization Pipeline for CryptoScalp AI

This module implements aggressive post-training optimization techniques to achieve
<1ms inference time with minimal accuracy loss for high-frequency trading.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, prepare, convert
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ModelPerformanceMetrics:
    """Container for model performance metrics"""

    def __init__(self):
        self.original_size_mb = 0
        self.optimized_size_mb = 0
        self.original_inference_time_ms = 0
        self.optimized_inference_time_ms = 0
        self.accuracy_drop = 0
        self.compression_ratio = 1.0
        self.speedup_ratio = 1.0


class AdvancedModelPruner:
    """
    Advanced model pruning with multiple strategies to maximize compression
    while maintaining accuracy.
    """

    def __init__(self):
        self.pruning_history = []

    def apply_structured_pruning(self, model: nn.Module, pruning_ratio: float = 0.3) -> nn.Module:
        """Apply structured pruning to reduce model size"""
        logger.info(f"Applying structured pruning with ratio {pruning_ratio}")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Prune entire neurons (structured pruning)
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                prune.remove(module, 'weight')

                # Also prune bias if it exists
                if hasattr(module, 'bias') and module.bias is not None:
                    prune.ln_structured(module, name='bias', amount=pruning_ratio, n=2, dim=0)
                    prune.remove(module, 'bias')

            elif isinstance(module, nn.Conv1d):
                # Prune entire filters for Conv1d
                prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                prune.remove(module, 'weight')

                if hasattr(module, 'bias') and module.bias is not None:
                    prune.ln_structured(module, name='bias', amount=pruning_ratio, n=2, dim=0)
                    prune.remove(module, 'bias')

        self.pruning_history.append({'type': 'structured', 'ratio': pruning_ratio})
        return model

    def apply_unstructured_pruning(self, model: nn.Module, pruning_ratio: float = 0.2) -> nn.Module:
        """Apply unstructured pruning for fine-grained compression"""
        logger.info(f"Applying unstructured pruning with ratio {pruning_ratio}")

        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_ratio
        )

        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        self.pruning_history.append({'type': 'unstructured', 'ratio': pruning_ratio})
        return model

    def apply_magnitude_pruning(self, model: nn.Module, threshold: float = 0.01) -> nn.Module:
        """Apply magnitude-based pruning"""
        logger.info(f"Applying magnitude pruning with threshold {threshold}")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # Get weight magnitude
                weight = module.weight.data.abs()
                mask = weight > threshold

                # Apply mask
                module.weight.data *= mask.float()

        self.pruning_history.append({'type': 'magnitude', 'threshold': threshold})
        return model


class AdvancedQuantizer:
    """
    Advanced quantization techniques for model optimization
    """

    def __init__(self):
        self.quantization_config = {}

    def apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization for LSTM and Linear layers"""
        logger.info("Applying dynamic quantization")

        # Specify which layers to quantize
        quantization_config = {
            nn.Linear: torch.qint8,
            nn.LSTM: torch.qint8,
            nn.Conv1d: torch.qint8
        }

        quantized_model = quantize_dynamic(
            model,
            quantization_config,
            dtype=torch.qint8
        )

        self.quantization_config['dynamic'] = True
        return quantized_model

    def apply_static_quantization(self, model: nn.Module,
                                calibration_data: torch.Tensor) -> nn.Module:
        """Apply static quantization with calibration"""
        logger.info("Applying static quantization")

        model.eval()

        # Prepare model for quantization
        model = prepare(model, inplace=False)

        # Calibration with representative data
        with torch.no_grad():
            for i in range(0, len(calibration_data), 32):
                batch = calibration_data[i:i+32]
                model(batch)

        # Convert to quantized model
        quantized_model = convert(model, inplace=False)

        self.quantization_config['static'] = True
        return quantized_model

    def apply_custom_quantization(self, model: nn.Module) -> nn.Module:
        """Apply custom quantization scheme optimized for trading signals"""
        logger.info("Applying custom quantization")

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Custom quantization for trading signal ranges
                # Direction: -1 to 1, Confidence: 0 to 1, Size: 0 to 1
                if 'signal' in name.lower() or 'output' in name.lower():
                    module.weight.data = self._quantize_weights_for_signals(module.weight.data)
                else:
                    module.weight.data = self._quantize_weights_symmetric(module.weight.data)

        self.quantization_config['custom'] = True
        return model

    def _quantize_weights_symmetric(self, weights: torch.Tensor,
                                  num_bits: int = 8) -> torch.Tensor:
        """Symmetric quantization"""
        max_val = weights.abs().max()
        scale = max_val / (2**(num_bits-1) - 1)
        quantized = torch.round(weights / scale)
        return torch.clamp(quantized * scale, -max_val, max_val)

    def _quantize_weights_for_signals(self, weights: torch.Tensor,
                                    num_bits: int = 8) -> torch.Tensor:
        """Optimized quantization for signal ranges"""
        # Different quantization for different output ranges
        # This is a simplified version - in practice would be more sophisticated
        scale = weights.abs().mean() * 2
        quantized = torch.round(weights / scale)
        return torch.clamp(quantized * scale, -1, 1)


class TensorRTOptimizer:
    """
    NVIDIA TensorRT optimization for maximum performance
    """

    def __init__(self):
        self.engine = None
        self.context = None

    def convert_to_tensorrt(self, model: nn.Module,
                          input_shape: Tuple[int, ...],
                          workspace_size: int = 1 << 30) -> Any:
        """Convert PyTorch model to TensorRT"""
        try:
            import tensorrt as trt
        except ImportError:
            logger.error("TensorRT not available. Install TensorRT for maximum performance.")
            return model

        logger.info("Converting model to TensorRT")

        # Create builder and network
        builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Convert to ONNX first
        onnx_path = Path("temp_model.onnx")
        self._export_to_onnx(model, input_shape, onnx_path)

        # Parse ONNX
        parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())

        # Build engine
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        # Enable optimizations
        config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)

        # Save engine
        engine_path = Path("optimized_model.engine")
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())

        self.engine = engine
        self.context = engine.create_execution_context()

        return engine

    def _export_to_onnx(self, model: nn.Module,
                       input_shape: Tuple[int, ...],
                       onnx_path: Path):
        """Export PyTorch model to ONNX format"""
        dummy_input = torch.randn(*input_shape)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )


class ModelOptimizationPipeline:
    """
    Complete model optimization pipeline
    """

    def __init__(self):
        self.pruner = AdvancedModelPruner()
        self.quantizer = AdvancedQuantizer()
        self.tensorrt_optimizer = TensorRTOptimizer()
        self.metrics = ModelPerformanceMetrics()

    def optimize_model(self, model: nn.Module,
                      calibration_data: Optional[torch.Tensor] = None,
                      target_inference_time_ms: float = 1.0) -> nn.Module:
        """Run complete optimization pipeline"""
        logger.info("Starting model optimization pipeline")

        # Step 1: Measure original performance
        self._measure_original_performance(model)

        # Step 2: Apply pruning
        model = self._apply_pruning_strategy(model)

        # Step 3: Apply quantization
        model = self._apply_quantization_strategy(model, calibration_data)

        # Step 4: Apply TensorRT optimization if available
        model = self._apply_tensorrt_optimization(model)

        # Step 5: Measure optimized performance
        self._measure_optimized_performance(model)

        # Step 6: Validate optimization quality
        self._validate_optimization()

        logger.info(f"Optimization complete. Speedup: {self.metrics.speedup_ratio:.2f}x, "
                   f"Compression: {self.metrics.compression_ratio:.2f}x")

        return model

    def _measure_original_performance(self, model: nn.Module):
        """Measure original model performance"""
        model.eval()

        # Measure model size
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        self.metrics.original_size_mb = param_size / 1024 / 1024

        # Measure inference time
        dummy_input = torch.randn(1, 1000)  # Typical input size
        times = []

        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)

            # Measure
            for _ in range(100):
                start = time.time()
                _ = model(dummy_input)
                times.append((time.time() - start) * 1000)

        self.metrics.original_inference_time_ms = np.mean(times)

    def _apply_pruning_strategy(self, model: nn.Module) -> nn.Module:
        """Apply intelligent pruning strategy"""
        # Start with moderate structured pruning
        model = self.pruner.apply_structured_pruning(model, pruning_ratio=0.2)

        # Follow with light unstructured pruning
        model = self.pruner.apply_unstructured_pruning(model, pruning_ratio=0.1)

        # Finish with magnitude-based pruning for fine-tuning
        model = self.pruner.apply_magnitude_pruning(model, threshold=0.005)

        return model

    def _apply_quantization_strategy(self, model: nn.Module,
                                  calibration_data: Optional[torch.Tensor]) -> nn.Module:
        """Apply quantization strategy based on available data"""
        if calibration_data is not None:
            # Use static quantization if calibration data available
            model = self.quantizer.apply_static_quantization(model, calibration_data)
        else:
            # Use dynamic quantization
            model = self.quantizer.apply_dynamic_quantization(model)

        # Always apply custom quantization for signal outputs
        model = self.quantizer.apply_custom_quantization(model)

        return model

    def _apply_tensorrt_optimization(self, model: nn.Module) -> nn.Module:
        """Apply TensorRT optimization if available"""
        try:
            input_shape = (1, 1000)
            optimized_model = self.tensorrt_optimizer.convert_to_tensorrt(
                model, input_shape
            )
            return optimized_model
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}")
            return model

    def _measure_optimized_performance(self, model: nn.Module):
        """Measure optimized model performance"""
        # Measure size
        if hasattr(model, 'parameters'):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            self.metrics.optimized_size_mb = param_size / 1024 / 1024

        # Measure inference time
        dummy_input = torch.randn(1, 1000)
        times = []

        try:
            with torch.no_grad():
                # Warmup
                for _ in range(10):
                    _ = model(dummy_input)

                # Measure
                for _ in range(100):
                    start = time.time()
                    _ = model(dummy_input)
                    times.append((time.time() - start) * 1000)

            self.metrics.optimized_inference_time_ms = np.mean(times)

        except Exception as e:
            logger.error(f"Performance measurement failed: {e}")
            self.metrics.optimized_inference_time_ms = self.metrics.original_inference_time_ms

    def _validate_optimization(self):
        """Validate optimization quality and compute metrics"""
        # Compute ratios
        self.metrics.compression_ratio = (
            self.metrics.original_size_mb / self.metrics.optimized_size_mb
            if self.metrics.optimized_size_mb > 0 else 1.0
        )

        self.metrics.speedup_ratio = (
            self.metrics.original_inference_time_ms / self.metrics.optimized_inference_time_ms
            if self.metrics.optimized_inference_time_ms > 0 else 1.0
        )

        # Log results
        logger.info(f"Model compression: {self.metrics.compression_ratio:.2f}x")
        logger.info(f"Inference speedup: {self.metrics.speedup_ratio:.2f}x")
        logger.info(f"Size reduction: {self.metrics.original_size_mb:.1f}MB -> "
                   f"{self.metrics.optimized_size_mb:.1f}MB")

    def save_optimization_report(self, path: str = "optimization_report.json"):
        """Save optimization report"""
        report = {
            'original_size_mb': self.metrics.original_size_mb,
            'optimized_size_mb': self.metrics.optimized_size_mb,
            'original_inference_time_ms': self.metrics.original_inference_time_ms,
            'optimized_inference_time_ms': self.metrics.optimized_inference_time_ms,
            'compression_ratio': self.metrics.compression_ratio,
            'speedup_ratio': self.metrics.speedup_ratio,
            'pruning_history': self.pruner.pruning_history,
            'quantization_config': self.quantizer.quantization_config
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Optimization report saved to {path}")


class OptimizedModelManager:
    """
    Manager for optimized models with automatic fallback
    """

    def __init__(self):
        self.original_model = None
        self.optimized_model = None
        self.use_optimized = True
        self.performance_threshold_ms = 1.0

    def set_models(self, original: nn.Module, optimized: nn.Module):
        """Set original and optimized models"""
        self.original_model = original
        self.optimized_model = optimized

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction with automatic fallback"""
        if self.use_optimized and self.optimized_model is not None:
            try:
                start_time = time.time()
                result = self.optimized_model(x)
                inference_time = (time.time() - start_time) * 1000

                # Check if performance is acceptable
                if inference_time <= self.performance_threshold_ms * 1.5:
                    return result
                else:
                    logger.warning(f"Optimized model too slow ({inference_time:.2f}ms), falling back")
                    self.use_optimized = False

            except Exception as e:
                logger.error(f"Optimized model failed: {e}, falling back")
                self.use_optimized = False

        # Fallback to original model
        if self.original_model is not None:
            return self.original_model(x)
        else:
            raise RuntimeError("No model available for prediction")

    def force_fallback(self, enable: bool = True):
        """Force fallback to original model"""
        self.use_optimized = not enable


# Example usage
if __name__ == "__main__":
    # Example model
    model = nn.Sequential(
        nn.Linear(1000, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 3)
    )

    # Optimization pipeline
    optimizer = ModelOptimizationPipeline()

    # Generate calibration data
    calibration_data = torch.randn(100, 1000)

    # Optimize model
    optimized_model = optimizer.optimize_model(
        model,
        calibration_data=calibration_data,
        target_inference_time_ms=1.0
    )

    # Save report
    optimizer.save_optimization_report()

    print("Model optimization complete!")
    print(f"Original size: {optimizer.metrics.original_size_mb:.1f}MB")
    print(f"Optimized size: {optimizer.metrics.optimized_size_mb:.1f}MB")
    print(f"Compression ratio: {optimizer.metrics.compression_ratio:.2f}x")
    print(f"Speedup ratio: {optimizer.metrics.speedup_ratio:.2f}x")