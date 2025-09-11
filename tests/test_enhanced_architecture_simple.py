"""
Simple Validation Script for Enhanced Architecture

This script performs basic validation of the enhanced architecture components
without requiring external dependencies or running complex ML operations.
"""

import sys
import os
import importlib.util
import inspect
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def validate_module_import(module_name: str, file_path: str) -> Dict[str, Any]:
    """Validate if a module can be imported and check its basic structure"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return {'success': False, 'error': 'Could not load module spec'}

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get all classes and functions
        classes = []
        functions = []

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and not name.startswith('_'):
                classes.append({
                    'name': name,
                    'methods': [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m))]
                })
            elif inspect.isfunction(obj) and not name.startswith('_'):
                functions.append(name)

        return {
            'success': True,
            'classes': classes,
            'functions': functions,
            'total_classes': len(classes),
            'total_functions': len(functions)
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def validate_class_structure(cls: Any) -> Dict[str, Any]:
    """Validate the structure of a class"""
    try:
        # Check if class has required methods
        methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m))]

        # Check if class can be instantiated (with basic parameters)
        try:
            # Try to create instance with default parameters
            if 'device' in inspect.signature(cls.__init__).parameters:
                instance = cls(device='cpu')
            elif len(inspect.signature(cls.__init__).parameters) == 1:
                instance = cls()
            else:
                # Can't instantiate without knowing parameters
                instance = None
        except Exception as e:
            instance = None

        return {
            'success': True,
            'methods_count': len(methods),
            'methods': methods,
            'can_instantiate': instance is not None,
            'instantiation_error': str(e) if instance is None else None
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}


def validate_architecture_components() -> Dict[str, Any]:
    """Validate all enhanced architecture components"""
    results = {}

    # Component validation mapping
    components = {
        'mixture_of_experts': 'src/models/mixture_of_experts.py',
        'model_optimizer': 'src/models/model_optimizer.py',
        'mlops_manager': 'src/models/mlops_manager.py',
        'self_awareness': 'src/models/self_awareness.py'
    }

    for component_name, file_path in components.items():
        if os.path.exists(file_path):
            print(f"Validating {component_name}...")

            # Test module import
            import_result = validate_module_import(component_name, file_path)

            if import_result['success']:
                results[component_name] = {
                    'import_success': True,
                    'classes_found': import_result['total_classes'],
                    'functions_found': import_result['total_functions'],
                    'class_details': import_result['classes']
                }

                # Test key classes if they can be imported
                try:
                    spec = importlib.util.spec_from_file_location(component_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Validate key classes for each component
                    if component_name == 'mixture_of_experts':
                        key_classes = ['MixtureOfExperts', 'MarketRegimeDetector', 'RegimeSpecificExpert']
                    elif component_name == 'model_optimizer':
                        key_classes = ['ModelOptimizationPipeline', 'AdvancedModelPruner', 'AdvancedQuantizer']
                    elif component_name == 'mlops_manager':
                        key_classes = ['FeatureStoreManager', 'EnhancedModelRegistry', 'AutomatedPipelineManager']
                    elif component_name == 'self_awareness':
                        key_classes = ['SelfAwarenessEngine', 'ExecutionStateTracker', 'AdaptiveBehaviorSystem']

                    class_validations = {}
                    for class_name in key_classes:
                        if hasattr(module, class_name):
                            cls = getattr(module, class_name)
                            class_validations[class_name] = validate_class_structure(cls)
                        else:
                            class_validations[class_name] = {'success': False, 'error': 'Class not found'}

                    results[component_name]['class_validations'] = class_validations

                except Exception as e:
                    results[component_name]['class_validation_error'] = str(e)

            else:
                results[component_name] = {
                    'import_success': False,
                    'error': import_result['error']
                }

        else:
            results[component_name] = {
                'import_success': False,
                'error': f'File not found: {file_path}'
            }

    return results


def validate_enhanced_features() -> Dict[str, Any]:
    """Validate that enhanced features are properly implemented"""
    features = {
        'moe_architecture': [
            'MarketRegimeDetector',
            'RegimeSpecificExpert',
            'MixtureOfExperts',
            'MoESignal',
            'RegimeClassification'
        ],
        'model_optimization': [
            'ModelOptimizationPipeline',
            'AdvancedModelPruner',
            'AdvancedQuantizer',
            'TensorRTOptimizer',
            'ModelPerformanceMetrics'
        ],
        'mlops_integration': [
            'FeatureStoreManager',
            'EnhancedModelRegistry',
            'AutomatedPipelineManager',
            'ModelMetadata',
            'FeatureMetadata'
        ],
        'self_awareness': [
            'SelfAwarenessEngine',
            'ExecutionStateTracker',
            'AdaptiveBehaviorSystem',
            'ExecutionEvent',
            'MarketImpactAnalyzer'
        ]
    }

    results = {}

    for feature_group, required_classes in features.items():
        results[feature_group] = {'found_classes': [], 'missing_classes': []}

        for class_name in required_classes:
            found = False

            # Check each component module
            for module_name in ['mixture_of_experts', 'model_optimizer', 'mlops_manager', 'self_awareness']:
                try:
                    module_path = f'src/models/{module_name}.py'
                    if os.path.exists(module_path):
                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        if hasattr(module, class_name):
                            results[feature_group]['found_classes'].append(class_name)
                            found = True
                            break
                except:
                    continue

            if not found:
                results[feature_group]['missing_classes'].append(class_name)

    return results


def generate_validation_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive validation report"""
    report = {
        'validation_timestamp': __import__('time').time(),
        'overall_success': True,
        'component_status': {},
        'feature_coverage': {},
        'recommendations': []
    }

    # Check component status
    for component, result in results['components'].items():
        if result.get('import_success', False):
            report['component_status'][component] = 'SUCCESS'
        else:
            report['component_status'][component] = 'FAILED'
            report['overall_success'] = False
            report['recommendations'].append(f"Fix import issues for {component}")

    # Check feature coverage
    total_required = 0
    total_found = 0

    for feature_group, feature_result in results['features'].items():
        required = len(feature_result.get('found_classes', [])) + len(feature_result.get('missing_classes', []))
        found = len(feature_result.get('found_classes', []))
        total_required += required
        total_found += found

        coverage = found / required if required > 0 else 0
        report['feature_coverage'][feature_group] = {
            'coverage_percentage': coverage * 100,
            'classes_found': found,
            'classes_required': required
        }

        if coverage < 0.8:  # Less than 80% coverage
            report['overall_success'] = False
            report['recommendations'].append(f"Improve feature coverage for {feature_group}")

    report['overall_coverage'] = total_found / total_required if total_required > 0 else 0

    return report


def main():
    """Main validation function"""
    print("Enhanced Architecture Validation")
    print("=" * 40)

    # Validate components
    component_results = validate_architecture_components()

    # Validate features
    feature_results = validate_enhanced_features()

    # Generate report
    validation_results = {
        'components': component_results,
        'features': feature_results
    }

    report = generate_validation_report(validation_results)

    # Print results
    print(f"\nOverall Status: {'PASS' if report['overall_success'] else 'FAIL'}")
    print(f"Overall Coverage: {report['overall_coverage']:.1%}")

    print(f"\nComponent Status:")
    for component, status in report['component_status'].items():
        print(f"  {component}: {status}")

    print(f"\nFeature Coverage:")
    for feature, coverage in report['feature_coverage'].items():
        print(f"  {feature}: {coverage['coverage_percentage']:.1f}% "
              f"({coverage['classes_found']}/{coverage['classes_required']} classes)")

    print(f"\nDetailed Component Results:")
    for component, result in component_results.items():
        if result.get('import_success'):
            print(f"  ✓ {component}: {result['classes_found']} classes, {result['functions_found']} functions")
        else:
            print(f"  ✗ {component}: {result.get('error', 'Unknown error')}")

    print(f"\nDetailed Feature Results:")
    for feature_group, result in feature_results.items():
        found = result.get('found_classes', [])
        missing = result.get('missing_classes', [])
        print(f"  {feature_group}:")
        if found:
            print(f"    ✓ Found: {', '.join(found)}")
        if missing:
            print(f"    ✗ Missing: {', '.join(missing)}")

    if report['recommendations']:
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

    print(f"\nValidation {'PASSED' if report['overall_success'] else 'FAILED'}")

    return report['overall_success']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)