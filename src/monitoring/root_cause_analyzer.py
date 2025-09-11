"""
Root Cause Analysis (RCA) Agent for CryptoScalp AI

This module implements advanced root cause analysis to identify the underlying causes
of system failures, performance degradation, and anomalies through correlation analysis,
log pattern recognition, and causal inference.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import numpy as np
import pandas as pd
from enum import Enum
import threading
import time
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Severity levels for incidents"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentType(Enum):
    """Types of incidents that can occur"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SYSTEM_FAILURE = "system_failure"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    TRADING_ERROR = "trading_error"
    INFRASTRUCTURE_ISSUE = "infrastructure_issue"
    SECURITY_INCIDENT = "security_incident"


@dataclass
class Incident:
    """Represents a system incident"""
    incident_id: str
    incident_type: IncidentType
    severity: Severity
    timestamp: datetime
    description: str
    affected_components: List[str]
    symptoms: List[str]
    raw_logs: List[str] = field(default_factory=list)
    metrics_data: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class RootCause:
    """Represents a root cause analysis result"""
    incident_id: str
    root_causes: List[str]
    contributing_factors: List[str]
    confidence: float
    evidence: List[str]
    recommended_actions: List[str]
    prevention_measures: List[str]
    analysis_timestamp: datetime
    analysis_duration_seconds: float


@dataclass
class CausalLink:
    """Represents a causal relationship between events"""
    source_event: str
    target_event: str
    correlation_strength: float
    time_delay_seconds: float
    evidence_count: int


class LogPatternAnalyzer:
    """Analyze log patterns to identify issues"""

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.warning_patterns = self._initialize_warning_patterns()

    def _initialize_error_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for error detection"""
        return {
            'memory_error': re.compile(r'(out of memory|memory allocation failed|heap exhausted)', re.IGNORECASE),
            'network_error': re.compile(r'(connection refused|timeout|network unreachable|dns resolution failed)', re.IGNORECASE),
            'database_error': re.compile(r'(connection pool exhausted|query failed|deadlock detected|database unavailable)', re.IGNORECASE),
            'model_error': re.compile(r'(model loading failed|inference error|tensor shape mismatch|cuda error)', re.IGNORECASE),
            'trading_error': re.compile(r'(order rejected|insufficient balance|market closed|rate limit exceeded)', re.IGNORECASE)
        }

    def _initialize_warning_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize regex patterns for warning detection"""
        return {
            'high_latency': re.compile(r'(latency > \d+ms|slow response|performance degradation)', re.IGNORECASE),
            'resource_usage': re.compile(r'(high cpu usage|memory leak suspected|disk space low)', re.IGNORECASE),
            'deprecation': re.compile(r'(deprecated|will be removed|use alternative)', re.IGNORECASE)
        }

    def analyze_logs(self, logs: List[str]) -> Dict[str, List[str]]:
        """Analyze log entries for patterns"""
        findings = defaultdict(list)

        for log_entry in logs:
            # Check error patterns
            for error_type, pattern in self.error_patterns.items():
                if pattern.search(log_entry):
                    findings[f"error_{error_type}"].append(log_entry)

            # Check warning patterns
            for warning_type, pattern in self.warning_patterns.items():
                if pattern.search(log_entry):
                    findings[f"warning_{warning_type}"].append(log_entry)

        return dict(findings)


class CorrelationAnalyzer:
    """Analyze correlations between metrics and events"""

    def __init__(self):
        self.correlation_threshold = 0.7
        self.min_samples = 10

    def analyze_metric_correlations(self, metrics_data: Dict[str, List[float]]) -> List[CausalLink]:
        """Analyze correlations between different metrics"""
        if len(metrics_data) < 2:
            return []

        causal_links = []
        metric_names = list(metrics_data.keys())

        # Calculate correlation matrix
        data_matrix = []
        for metric in metric_names:
            data_matrix.append(metrics_data[metric][-self.min_samples:])

        try:
            correlation_matrix = np.corrcoef(data_matrix)

            # Find strong correlations
            for i in range(len(metric_names)):
                for j in range(i+1, len(metric_names)):
                    correlation = correlation_matrix[i, j]

                    if abs(correlation) >= self.correlation_threshold:
                        # Calculate time delay (simplified)
                        time_delay = self._estimate_time_delay(
                            data_matrix[i], data_matrix[j]
                        )

                        causal_link = CausalLink(
                            source_event=f"metric_{metric_names[i]}",
                            target_event=f"metric_{metric_names[j]}",
                            correlation_strength=correlation,
                            time_delay_seconds=time_delay,
                            evidence_count=self.min_samples
                        )

                        causal_links.append(causal_link)

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")

        return causal_links

    def _estimate_time_delay(self, series1: List[float], series2: List[float]) -> float:
        """Estimate time delay between two series (simplified)"""
        # This is a simplified implementation
        # In practice, you'd use cross-correlation or more sophisticated methods
        max_delay = min(5, len(series1) // 2)  # Check up to 5 time steps

        best_delay = 0
        best_correlation = 0

        for delay in range(max_delay + 1):
            if delay >= len(series2):
                continue

            correlation = np.corrcoef(series1[:-delay], series2[delay:])[0, 1]
            if abs(correlation) > abs(best_correlation):
                best_correlation = correlation
                best_delay = delay

        return best_delay * 60  # Convert to seconds (assuming 1-minute intervals)


class CausalInferenceEngine:
    """Perform causal inference to identify root causes"""

    def __init__(self):
        self.causal_graph = self._build_causal_graph()

    def _build_causal_graph(self) -> Dict[str, List[str]]:
        """Build a causal graph of system components"""
        return {
            'high_cpu_usage': ['slow_response_times', 'failed_requests'],
            'memory_leak': ['out_of_memory_errors', 'system_crashes'],
            'network_latency': ['timeout_errors', 'failed_connections'],
            'database_connection_issues': ['query_failures', 'data_unavailability'],
            'model_loading_failures': ['inference_errors', 'trading_halt'],
            'data_quality_issues': ['incorrect_predictions', 'trading_losses'],
            'market_volatility': ['model_performance_degradation', 'increased_risk']
        }

    def infer_causes(self, symptoms: List[str], evidence: Dict[str, Any]) -> List[str]:
        """Infer root causes from symptoms and evidence"""
        potential_causes = []

        # Direct mapping from symptoms to causes
        for symptom in symptoms:
            symptom_lower = symptom.lower()

            for cause, effects in self.causal_graph.items():
                if any(effect in symptom_lower for effect in effects):
                    potential_causes.append(cause)

        # Evidence-based inference
        if 'error_patterns' in evidence:
            error_patterns = evidence['error_patterns']
            if 'error_memory_error' in error_patterns:
                potential_causes.append('memory_leak')
            if 'error_network_error' in error_patterns:
                potential_causes.append('network_latency')
            if 'error_database_error' in error_patterns:
                potential_causes.append('database_connection_issues')

        # Metric-based inference
        if 'metrics_data' in evidence:
            metrics = evidence['metrics_data']
            if 'cpu_usage' in metrics and metrics['cpu_usage'][-1] > 90:
                potential_causes.append('high_cpu_usage')
            if 'memory_usage' in metrics and metrics['memory_usage'][-1] > 85:
                potential_causes.append('memory_leak')

        return list(set(potential_causes))  # Remove duplicates


class RootCauseAnalysisAgent:
    """
    Advanced root cause analysis agent that combines multiple analysis techniques
    to identify the underlying causes of system incidents.
    """

    def __init__(self):
        self.log_analyzer = LogPatternAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.causal_inference = CausalInferenceEngine()

        # Analysis history
        self.analysis_history: Dict[str, RootCause] = {}
        self.incident_patterns: Dict[str, int] = defaultdict(int)

        # Thread safety
        self._lock = threading.Lock()

    async def analyze_incident(self, incident: Incident) -> RootCause:
        """
        Perform comprehensive root cause analysis on an incident.
        This is the main method that coordinates all analysis techniques.
        """
        analysis_start = time.time()

        logger.info(f"Starting root cause analysis for incident {incident.incident_id}")

        # 1. Analyze log patterns
        log_findings = self.log_analyzer.analyze_logs(incident.raw_logs)

        # 2. Analyze metric correlations
        causal_links = self.correlation_analyzer.analyze_metric_correlations(
            incident.metrics_data
        )

        # 3. Perform causal inference
        evidence = {
            'error_patterns': log_findings,
            'metrics_data': incident.metrics_data,
            'causal_links': causal_links,
            'affected_components': incident.affected_components
        }

        root_causes = self.causal_inference.infer_causes(incident.symptoms, evidence)

        # 4. Identify contributing factors
        contributing_factors = await self._identify_contributing_factors(
            incident, log_findings, causal_links
        )

        # 5. Generate recommendations
        recommended_actions = self._generate_recommended_actions(
            root_causes, incident.incident_type
        )

        # 6. Generate prevention measures
        prevention_measures = self._generate_prevention_measures(
            root_causes, incident.incident_type
        )

        # 7. Calculate confidence
        confidence = self._calculate_analysis_confidence(
            log_findings, causal_links, root_causes
        )

        # 8. Generate evidence
        evidence_list = self._compile_evidence(
            incident, log_findings, causal_links, root_causes
        )

        analysis_duration = time.time() - analysis_start

        root_cause = RootCause(
            incident_id=incident.incident_id,
            root_causes=root_causes,
            contributing_factors=contributing_factors,
            confidence=confidence,
            evidence=evidence_list,
            recommended_actions=recommended_actions,
            prevention_measures=prevention_measures,
            analysis_timestamp=datetime.now(),
            analysis_duration_seconds=analysis_duration
        )

        # Store in history
        with self._lock:
            self.analysis_history[incident.incident_id] = root_cause
            self._update_incident_patterns(incident, root_causes)

        logger.info(f"Completed root cause analysis for {incident.incident_id} "
                   f"in {analysis_duration:.2f}s with {len(root_causes)} root causes")

        return root_cause

    async def _identify_contributing_factors(self, incident: Incident,
                                          log_findings: Dict[str, List[str]],
                                          causal_links: List[CausalLink]) -> List[str]:
        """Identify contributing factors to the incident"""
        contributing_factors = []

        # Analyze log patterns for contributing factors
        if 'error_memory_error' in log_findings:
            contributing_factors.append("Memory pressure from concurrent operations")
        if 'error_network_error' in log_findings:
            contributing_factors.append("Network connectivity issues")

        # Analyze causal links
        for link in causal_links:
            if link.correlation_strength > 0.8:
                contributing_factors.append(
                    f"Strong correlation between {link.source_event} and {link.target_event} "
                    f"(delay: {link.time_delay_seconds}s)"
                )

        # Component-specific factors
        for component in incident.affected_components:
            if 'database' in component.lower():
                contributing_factors.append("Database connection pool exhaustion")
            elif 'model' in component.lower():
                contributing_factors.append("Model inference pipeline bottleneck")
            elif 'trading' in component.lower():
                contributing_factors.append("High-frequency trading load")

        return contributing_factors

    def _generate_recommended_actions(self, root_causes: List[str],
                                    incident_type: IncidentType) -> List[str]:
        """Generate recommended actions based on root causes"""
        actions = []

        for cause in root_causes:
            if 'memory' in cause.lower():
                actions.extend([
                    "Increase memory allocation",
                    "Implement memory leak detection",
                    "Add memory usage monitoring alerts"
                ])
            elif 'cpu' in cause.lower():
                actions.extend([
                    "Scale up CPU resources",
                    "Optimize code performance",
                    "Implement load balancing"
                ])
            elif 'network' in cause.lower():
                actions.extend([
                    "Review network configuration",
                    "Implement retry logic with exponential backoff",
                    "Add network monitoring"
                ])
            elif 'database' in cause.lower():
                actions.extend([
                    "Optimize database queries",
                    "Increase connection pool size",
                    "Implement database connection health checks"
                ])

        # Incident-type specific actions
        if incident_type == IncidentType.PERFORMANCE_DEGRADATION:
            actions.append("Implement performance monitoring and alerting")
        elif incident_type == IncidentType.TRADING_ERROR:
            actions.append("Add additional validation layers for trading operations")

        return list(set(actions))  # Remove duplicates

    def _generate_prevention_measures(self, root_causes: List[str],
                                   incident_type: IncidentType) -> List[str]:
        """Generate prevention measures"""
        measures = []

        for cause in root_causes:
            if 'memory' in cause.lower():
                measures.extend([
                    "Implement regular memory usage monitoring",
                    "Set up automated memory leak detection",
                    "Configure memory usage alerts"
                ])
            elif 'cpu' in cause.lower():
                measures.extend([
                    "Implement auto-scaling based on CPU usage",
                    "Regular performance optimization reviews",
                    "Load testing in staging environment"
                ])

        # Add general prevention measures
        measures.extend([
            "Improve monitoring and alerting coverage",
            "Implement automated incident response",
            "Regular system health checks",
            "Documentation of incident response procedures"
        ])

        return list(set(measures))

    def _calculate_analysis_confidence(self, log_findings: Dict[str, List[str]],
                                    causal_links: List[CausalLink],
                                    root_causes: List[str]) -> float:
        """Calculate confidence score for the analysis"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on evidence
        if log_findings:
            confidence += 0.1 * min(len(log_findings), 5)  # Up to 0.5 for log evidence

        if causal_links:
            confidence += 0.1 * min(len(causal_links), 3)  # Up to 0.3 for correlations

        if root_causes:
            confidence += 0.1 * min(len(root_causes), 2)  # Up to 0.2 for identified causes

        return min(confidence, 0.95)  # Cap at 95%

    def _compile_evidence(self, incident: Incident, log_findings: Dict[str, List[str]],
                         causal_links: List[CausalLink], root_causes: List[str]) -> List[str]:
        """Compile evidence supporting the root cause analysis"""
        evidence = []

        # Log-based evidence
        for finding_type, logs in log_findings.items():
            if logs:
                evidence.append(f"Found {len(logs)} instances of {finding_type} in logs")

        # Correlation-based evidence
        for link in causal_links:
            evidence.append(f"Strong correlation ({link.correlation_strength:.2f}) "
                          f"between {link.source_event} and {link.target_event}")

        # Component-based evidence
        evidence.append(f"Incident affected {len(incident.affected_components)} components: "
                       f"{', '.join(incident.affected_components)}")

        # Metric-based evidence
        if incident.metrics_data:
            evidence.append(f"Analysis based on {len(incident.metrics_data)} metrics")

        return evidence

    def _update_incident_patterns(self, incident: Incident, root_causes: List[str]):
        """Update incident pattern tracking for learning"""
        pattern_key = f"{incident.incident_type.value}_{'_'.join(root_causes[:2])}"
        self.incident_patterns[pattern_key] += 1

    def get_common_root_causes(self, incident_type: Optional[IncidentType] = None) -> List[Tuple[str, int]]:
        """Get most common root causes"""
        if incident_type:
            prefix = f"{incident_type.value}_"
            relevant_patterns = {k: v for k, v in self.incident_patterns.items()
                               if k.startswith(prefix)}
        else:
            relevant_patterns = self.incident_patterns

        return Counter(relevant_patterns).most_common(10)

    def get_analysis_history(self, incident_id: Optional[str] = None) -> Dict[str, RootCause]:
        """Get analysis history"""
        if incident_id:
            return {incident_id: self.analysis_history.get(incident_id)} if incident_id in self.analysis_history else {}
        return self.analysis_history.copy()

    def save_analysis_report(self, incident_id: str, path: str):
        """Save analysis report to file"""
        analysis = self.analysis_history.get(incident_id)
        if not analysis:
            raise ValueError(f"No analysis found for incident {incident_id}")

        report = {
            'incident_id': analysis.incident_id,
            'root_causes': analysis.root_causes,
            'contributing_factors': analysis.contributing_factors,
            'confidence': analysis.confidence,
            'evidence': analysis.evidence,
            'recommended_actions': analysis.recommended_actions,
            'prevention_measures': analysis.prevention_measures,
            'analysis_timestamp': analysis.analysis_timestamp.isoformat(),
            'analysis_duration_seconds': analysis.analysis_duration_seconds
        }

        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Saved analysis report to {path}")

    def get_system_health_insights(self) -> Dict[str, Any]:
        """Get insights about system health based on analysis history"""
        insights = {
            'total_analyses': len(self.analysis_history),
            'common_root_causes': self.get_common_root_causes(),
            'average_confidence': 0.0,
            'most_affected_components': self._get_most_affected_components()
        }

        if self.analysis_history:
            confidences = [analysis.confidence for analysis in self.analysis_history.values()]
            insights['average_confidence'] = sum(confidences) / len(confidences)

        return insights

    def _get_most_affected_components(self) -> List[Tuple[str, int]]:
        """Get components most frequently affected by incidents"""
        component_counts = defaultdict(int)

        for analysis in self.analysis_history.values():
            # This would need access to the original incident data
            # For now, return a placeholder
            pass

        return []


# Example usage and testing
if __name__ == "__main__":
    async def test_root_cause_analysis():
        # Create RCA agent
        rca_agent = RootCauseAnalysisAgent()

        # Create a mock incident
        incident = Incident(
            incident_id="incident_001",
            incident_type=IncidentType.PERFORMANCE_DEGRADATION,
            severity=Severity.HIGH,
            timestamp=datetime.now(),
            description="Trading system response times increased by 300%",
            affected_components=["trading_engine", "model_inference"],
            symptoms=[
                "Slow response times",
                "High CPU usage",
                "Memory usage increasing",
                "Failed requests"
            ],
            raw_logs=[
                "2024-01-01 10:00:00 ERROR Memory allocation failed in trading_engine",
                "2024-01-01 10:01:00 WARNING High CPU usage: 95%",
                "2024-01-01 10:02:00 ERROR Network timeout after 30 seconds"
            ],
            metrics_data={
                'cpu_usage': [85, 87, 90, 92, 95],
                'memory_usage': [70, 75, 80, 82, 85],
                'response_time': [100, 120, 150, 200, 300]
            }
        )

        # Analyze the incident
        root_cause = await rca_agent.analyze_incident(incident)

        print(f"Root Cause Analysis for {incident.incident_id}")
        print(f"Confidence: {root_cause.confidence:.2f}")
        print(f"Root Causes: {root_cause.root_causes}")
        print(f"Contributing Factors: {root_cause.contributing_factors}")
        print(f"Recommended Actions: {root_cause.recommended_actions}")

        # Save report
        rca_agent.save_analysis_report(incident.incident_id, "rca_report.json")

        # Get insights
        insights = rca_agent.get_system_health_insights()
        print(f"System Health Insights: {insights}")

    # Run test
    asyncio.run(test_root_cause_analysis())