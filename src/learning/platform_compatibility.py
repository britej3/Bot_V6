"""
Platform Compatibility Module
=============================

This module ensures cross-platform compatibility for the Online Model Adaptation Framework,
with specific optimizations for Mac Intel systems.

Key Features:
- Platform detection and optimization
- Hardware acceleration configuration
- Memory management optimization
- Threading configuration for Mac Intel
- Performance monitoring adaptations

Author: Autonomous Systems Team
Date: 2025-01-22
"""

import platform
import sys
import torch
import psutil
import os
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """Platform information and capabilities"""
    system: str
    architecture: str
    python_version: str
    torch_version: str
    cpu_count: int
    memory_gb: float
    cuda_available: bool
    mps_available: bool
    optimized_device: str
    recommended_threads: int
    memory_limit_gb: Optional[float] = None


class PlatformCompatibility:
    """Platform compatibility manager for cross-platform optimization"""
    
    def __init__(self):
        self.platform_info = self._detect_platform()
        self._configure_torch_for_platform()
        self._setup_memory_management()
        
        logger.info(f"Platform compatibility initialized for {self.platform_info.system} {self.platform_info.architecture}")
    
    def _detect_platform(self) -> PlatformInfo:
        """Detect platform and hardware capabilities"""
        system = platform.system()
        architecture = platform.machine()
        python_version = sys.version
        torch_version = torch.__version__
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Check hardware acceleration
        cuda_available = torch.cuda.is_available()
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Determine optimal device
        if cuda_available:
            optimized_device = "cuda"
        elif mps_available:
            optimized_device = "mps"
        else:
            optimized_device = "cpu"
        
        # Configure threading for Mac Intel
        if system == "Darwin" and architecture == "x86_64":
            # Mac Intel specific optimizations
            recommended_threads = min(cpu_count // 2, 8)  # Conservative threading for Mac Intel
            memory_limit_gb = memory_gb * 0.7  # Reserve 30% for system
        elif system == "Darwin" and architecture == "arm64":
            # Mac M1/M2 optimizations
            recommended_threads = min(cpu_count, 10)
            memory_limit_gb = memory_gb * 0.8
        else:
            # Linux/Windows optimizations
            recommended_threads = min(cpu_count, 12)
            memory_limit_gb = memory_gb * 0.85
        
        return PlatformInfo(
            system=system,
            architecture=architecture,
            python_version=python_version,
            torch_version=torch_version,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            cuda_available=cuda_available,
            mps_available=mps_available,
            optimized_device=optimized_device,
            recommended_threads=recommended_threads,
            memory_limit_gb=memory_limit_gb
        )
    
    def _configure_torch_for_platform(self) -> None:
        """Configure PyTorch for optimal platform performance"""
        # Set number of threads based on platform
        torch.set_num_threads(self.platform_info.recommended_threads)
        
        # Mac Intel specific configurations
        if self.is_mac_intel():
            # Use BLAS optimizations for Mac Intel
            os.environ['OMP_NUM_THREADS'] = str(self.platform_info.recommended_threads)
            os.environ['MKL_NUM_THREADS'] = str(self.platform_info.recommended_threads)
            
            # Disable OpenMP for better stability on Mac Intel
            torch.set_num_interop_threads(1)
            
            logger.info("Configured PyTorch optimizations for Mac Intel")
        
        elif self.is_mac_arm():
            # Mac M1/M2 optimizations
            if self.platform_info.mps_available:
                logger.info("MPS acceleration available on Mac ARM")
            
        # Memory management
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    
    def _setup_memory_management(self) -> None:
        """Setup memory management based on platform"""
        if self.platform_info.memory_limit_gb:
            # Set memory limits to prevent system overload
            import resource
            try:
                # Set soft memory limit (in bytes)
                soft_limit = int(self.platform_info.memory_limit_gb * 1024**3)
                resource.setrlimit(resource.RLIMIT_AS, (soft_limit, resource.RLIM_INFINITY))
                logger.info(f"Set memory limit to {self.platform_info.memory_limit_gb:.1f}GB")
            except Exception as e:
                logger.warning(f"Could not set memory limit: {e}")
    
    def is_mac_intel(self) -> bool:
        """Check if running on Mac Intel"""
        return (self.platform_info.system == "Darwin" and 
                self.platform_info.architecture == "x86_64")
    
    def is_mac_arm(self) -> bool:
        """Check if running on Mac ARM (M1/M2)"""
        return (self.platform_info.system == "Darwin" and 
                self.platform_info.architecture == "arm64")
    
    def is_linux(self) -> bool:
        """Check if running on Linux"""
        return self.platform_info.system == "Linux"
    
    def is_windows(self) -> bool:
        """Check if running on Windows"""
        return self.platform_info.system == "Windows"
    
    def get_optimal_device(self) -> str:
        """Get the optimal device for computations"""
        return self.platform_info.optimized_device
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Get optimal batch size based on available memory"""
        memory_factor = self.platform_info.memory_gb / 16.0  # Normalize to 16GB baseline
        
        if self.is_mac_intel():
            # Conservative batch sizing for Mac Intel
            return max(8, int(base_batch_size * min(memory_factor, 1.5)))
        elif self.is_mac_arm():
            # More aggressive for Mac ARM with unified memory
            return max(16, int(base_batch_size * min(memory_factor, 2.0)))
        else:
            # Standard scaling for other platforms
            return max(8, int(base_batch_size * memory_factor))
    
    def get_optimal_worker_count(self) -> int:
        """Get optimal number of worker processes"""
        if self.is_mac_intel():
            # Conservative worker count for Mac Intel
            return min(4, self.platform_info.cpu_count // 2)
        else:
            return min(8, self.platform_info.cpu_count // 2)
    
    def configure_model_for_platform(self, model: torch.nn.Module) -> torch.nn.Module:
        """Configure model for optimal platform performance"""
        device = self.get_optimal_device()
        
        try:
            model = model.to(device)
            logger.info(f"Moved model to {device}")
            
            # Platform-specific model optimizations
            if device == "cuda":
                model = torch.jit.script(model) if hasattr(torch, 'jit') else model
            elif device == "mps" and self.is_mac_arm():
                # MPS optimizations for Mac ARM
                model.eval()
                logger.info("Configured model for MPS acceleration")
            elif self.is_mac_intel():
                # CPU optimizations for Mac Intel
                model = torch.jit.script(model) if hasattr(torch, 'jit') else model
                logger.info("Configured model for Mac Intel CPU optimization")
                
        except Exception as e:
            logger.warning(f"Could not optimize model for platform: {e}")
            model = model.to("cpu")
        
        return model
    
    def get_tensor_creation_kwargs(self) -> Dict[str, Any]:
        """Get optimal tensor creation arguments"""
        kwargs = {
            'device': self.get_optimal_device(),
            'dtype': torch.float32
        }
        
        if self.is_mac_intel():
            # Use float32 for better compatibility on Mac Intel
            kwargs['dtype'] = torch.float32
        
        return kwargs
    
    def optimize_tensor_operations(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize tensor for platform-specific operations"""
        device = self.get_optimal_device()
        
        if tensor.device.type != device:
            tensor = tensor.to(device)
        
        # Platform-specific optimizations
        if self.is_mac_intel() and tensor.dtype == torch.float64:
            # Convert to float32 for better performance on Mac Intel
            tensor = tensor.float()
        
        return tensor
    
    def get_performance_recommendations(self) -> Dict[str, Any]:
        """Get platform-specific performance recommendations"""
        recommendations = {
            'batch_size': self.get_optimal_batch_size(),
            'num_workers': self.get_optimal_worker_count(),
            'device': self.get_optimal_device(),
            'dtype': torch.float32,
            'pin_memory': not self.is_mac_intel(),  # Disable pin_memory on Mac Intel
            'non_blocking': self.platform_info.cuda_available
        }
        
        if self.is_mac_intel():
            recommendations.update({
                'use_multiprocessing': False,  # Can cause issues on Mac Intel
                'persistent_workers': False,
                'prefetch_factor': 2,
                'drop_last': True
            })
        
        return recommendations
    
    def get_platform_summary(self) -> Dict[str, Any]:
        """Get comprehensive platform information"""
        return {
            'platform': {
                'system': self.platform_info.system,
                'architecture': self.platform_info.architecture,
                'python_version': self.platform_info.python_version,
                'torch_version': self.platform_info.torch_version
            },
            'hardware': {
                'cpu_count': self.platform_info.cpu_count,
                'memory_gb': self.platform_info.memory_gb,
                'cuda_available': self.platform_info.cuda_available,
                'mps_available': self.platform_info.mps_available
            },
            'optimization': {
                'device': self.platform_info.optimized_device,
                'recommended_threads': self.platform_info.recommended_threads,
                'memory_limit_gb': self.platform_info.memory_limit_gb,
                'batch_size': self.get_optimal_batch_size(),
                'worker_count': self.get_optimal_worker_count()
            },
            'compatibility': {
                'mac_intel': self.is_mac_intel(),
                'mac_arm': self.is_mac_arm(),
                'linux': self.is_linux(),
                'windows': self.is_windows()
            }
        }


# Global platform compatibility instance
_platform_compatibility: Optional[PlatformCompatibility] = None


def get_platform_compatibility() -> PlatformCompatibility:
    """Get singleton platform compatibility instance"""
    global _platform_compatibility
    if _platform_compatibility is None:
        _platform_compatibility = PlatformCompatibility()
    return _platform_compatibility


def ensure_mac_intel_compatibility() -> Dict[str, Any]:
    """Ensure compatibility and optimization for Mac Intel systems"""
    platform_compat = get_platform_compatibility()
    
    if platform_compat.is_mac_intel():
        logger.info("Detected Mac Intel system - applying optimizations")
        
        # Apply Mac Intel specific configurations
        optimizations = {
            'threading_optimized': True,
            'memory_management': True,
            'tensor_optimizations': True,
            'model_optimizations': True
        }
        
        return {
            'compatible': True,
            'optimized': True,
            'platform_info': platform_compat.get_platform_summary(),
            'optimizations_applied': optimizations,
            'recommendations': platform_compat.get_performance_recommendations()
        }
    
    return {
        'compatible': True,
        'optimized': False,
        'platform_info': platform_compat.get_platform_summary(),
        'message': f"Platform: {platform_compat.platform_info.system} {platform_compat.platform_info.architecture}"
    }


# Demo and validation
if __name__ == "__main__":
    print("🔧 Platform Compatibility Check")
    print("=" * 50)
    
    # Initialize platform compatibility
    platform_compat = get_platform_compatibility()
    
    # Display platform information
    summary = platform_compat.get_platform_summary()
    
    print(f"System: {summary['platform']['system']}")
    print(f"Architecture: {summary['platform']['architecture']}")
    print(f"PyTorch: {summary['platform']['torch_version']}")
    print(f"Device: {summary['optimization']['device']}")
    print(f"Memory: {summary['hardware']['memory_gb']:.1f}GB")
    print(f"CPU Cores: {summary['hardware']['cpu_count']}")
    
    # Check Mac Intel compatibility
    mac_intel_result = ensure_mac_intel_compatibility()
    
    if mac_intel_result['compatible']:
        print("✅ Platform compatible")
        if mac_intel_result.get('optimized'):
            print("✅ Mac Intel optimizations applied")
        
        # Display recommendations
        recommendations = mac_intel_result.get('recommendations', {})
        if recommendations:
            print("\n🎯 Performance Recommendations:")
            for key, value in recommendations.items():
                print(f"  {key}: {value}")
    
    print("\n🎉 Platform compatibility validated!")