"""
Recovery Playbook Repository for CryptoScalp AI

This module implements a comprehensive repository of automated recovery playbooks
that can be executed to restore system functionality after failures. Includes both
reactive playbooks (triggered by detected issues) and proactive playbooks (triggered
by predicted failures).
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import yaml
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class PlaybookType(Enum):
    """Types of recovery playbooks"""
    REACTIVE = "reactive"
    PROACTIVE = "proactive"


class PlaybookStatus(Enum):
    """Execution status of playbooks"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PlaybookStep:
    """Individual step in a recovery playbook"""
    name: str
    description: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    required_success: bool = True
    rollback_action: Optional[str] = None


@dataclass
class RecoveryPlaybook:
    """Complete recovery playbook"""
    playbook_id: str
    name: str
    description: str
    playbook_type: PlaybookType
    trigger_conditions: List[str]
    steps: List[PlaybookStep]
    estimated_duration_seconds: int = 600
    risk_level: str = "medium"  # 'low', 'medium', 'high'
    requires_approval: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass
class PlaybookExecution:
    """Record of a playbook execution"""
    execution_id: str
    playbook_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: PlaybookStatus = PlaybookStatus.PENDING
    current_step: int = 0
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    rollback_actions: List[str] = field(default_factory=list)


class PlaybookExecutor:
    """Executes recovery playbooks"""

    def __init__(self):
        self.action_handlers: Dict[str, Callable] = {}
        self.register_default_actions()

    def register_action_handler(self, action_name: str, handler: Callable):
        """Register a handler for a specific action"""
        self.action_handlers[action_name] = handler
        logger.info(f"Registered action handler: {action_name}")

    def register_default_actions(self):
        """Register default action handlers"""
        self.register_action_handler("restart_service", self._restart_service)
        self.register_action_handler("scale_resources", self._scale_resources)
        self.register_action_handler("switch_to_backup", self._switch_to_backup)
        self.register_action_handler("run_health_check", self._run_health_check)
        self.register_action_handler("alert_team", self._alert_team)
        self.register_action_handler("rollback_deployment", self._rollback_deployment)
        self.register_action_handler("clear_cache", self._clear_cache)
        self.register_action_handler("optimize_database", self._optimize_database)
        self.register_action_handler("reconnect_network", self._reconnect_network)
        self.register_action_handler("increase_memory", self._increase_memory)

    async def execute_playbook(self, playbook: RecoveryPlaybook,
                             context: Dict[str, Any] = None) -> PlaybookExecution:
        """Execute a recovery playbook"""
        execution = PlaybookExecution(
            execution_id=f"exec_{playbook.playbook_id}_{int(time.time())}",
            playbook_id=playbook.playbook_id,
            start_time=datetime.now()
        )

        logger.info(f"Starting playbook execution: {playbook.name} ({execution.execution_id})")

        try:
            execution.status = PlaybookStatus.RUNNING

            for i, step in enumerate(playbook.steps):
                execution.current_step = i
                await self._execute_step(step, execution, context or {})

                # Check if step failed and is required
                if execution.status == PlaybookStatus.FAILED and step.required_success:
                    logger.error(f"Required step failed: {step.name}")
                    await self._execute_rollback_actions(playbook, execution)
                    break

            if execution.status == PlaybookStatus.RUNNING:
                execution.status = PlaybookStatus.COMPLETED
                logger.info(f"Playbook completed successfully: {playbook.name}")

        except Exception as e:
            logger.error(f"Playbook execution failed: {e}")
            execution.status = PlaybookStatus.FAILED
            execution.errors.append(str(e))
            await self._execute_rollback_actions(playbook, execution)

        execution.end_time = datetime.now()
        return execution

    async def _execute_step(self, step: PlaybookStep, execution: PlaybookExecution,
                          context: Dict[str, Any]):
        """Execute a single playbook step"""
        logger.info(f"Executing step: {step.name}")

        try:
            if step.action in self.action_handlers:
                handler = self.action_handlers[step.action]

                # Execute with timeout
                await asyncio.wait_for(
                    handler(step.parameters, context),
                    timeout=step.timeout_seconds
                )

                execution.results[step.name] = "success"
                logger.info(f"Step completed: {step.name}")
            else:
                raise ValueError(f"No handler registered for action: {step.action}")

        except asyncio.TimeoutError:
            error_msg = f"Step timeout after {step.timeout_seconds}s: {step.name}"
            execution.errors.append(error_msg)
            execution.status = PlaybookStatus.FAILED
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Step failed: {step.name} - {str(e)}"
            execution.errors.append(error_msg)
            execution.status = PlaybookStatus.FAILED
            logger.error(error_msg)

    async def _execute_rollback_actions(self, playbook: RecoveryPlaybook,
                                     execution: PlaybookExecution):
        """Execute rollback actions for failed playbook"""
        logger.info("Executing rollback actions")

        for step in playbook.steps[:execution.current_step + 1]:
            if step.rollback_action:
                try:
                    if step.rollback_action in self.action_handlers:
                        handler = self.action_handlers[step.rollback_action]
                        await handler(step.parameters, {})
                        execution.rollback_actions.append(f"Rolled back: {step.name}")
                        logger.info(f"Rollback completed: {step.name}")
                    else:
                        execution.errors.append(f"No rollback handler for: {step.rollback_action}")
                except Exception as e:
                    execution.errors.append(f"Rollback failed for {step.name}: {str(e)}")
                    logger.error(f"Rollback failed: {step.name} - {str(e)}")

    # Default action handlers
    async def _restart_service(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Restart a service"""
        service_name = params.get('service_name', 'unknown')
        logger.info(f"Restarting service: {service_name}")
        await asyncio.sleep(2)  # Simulate restart time

    async def _scale_resources(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Scale resources"""
        resource_type = params.get('resource_type', 'cpu')
        scale_factor = params.get('scale_factor', 1.5)
        logger.info(f"Scaling {resource_type} by factor {scale_factor}")
        await asyncio.sleep(1)

    async def _switch_to_backup(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Switch to backup system"""
        system_name = params.get('system_name', 'backup')
        logger.info(f"Switching to backup system: {system_name}")
        await asyncio.sleep(3)

    async def _run_health_check(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Run health check"""
        component = params.get('component', 'system')
        logger.info(f"Running health check for: {component}")
        await asyncio.sleep(1)

    async def _alert_team(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Alert the team"""
        urgency = params.get('urgency', 'medium')
        message = params.get('message', 'System alert')
        logger.info(f"Alerting team ({urgency}): {message}")
        await asyncio.sleep(0.5)

    async def _rollback_deployment(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Rollback deployment"""
        version = params.get('version', 'previous')
        logger.info(f"Rolling back to version: {version}")
        await asyncio.sleep(5)

    async def _clear_cache(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Clear cache"""
        cache_type = params.get('cache_type', 'all')
        logger.info(f"Clearing cache: {cache_type}")
        await asyncio.sleep(1)

    async def _optimize_database(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Optimize database"""
        operation = params.get('operation', 'reindex')
        logger.info(f"Optimizing database: {operation}")
        await asyncio.sleep(10)

    async def _reconnect_network(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Reconnect network"""
        interface = params.get('interface', 'default')
        logger.info(f"Reconnecting network interface: {interface}")
        await asyncio.sleep(2)

    async def _increase_memory(self, params: Dict[str, Any], context: Dict[str, Any]):
        """Increase memory allocation"""
        increase_gb = params.get('increase_gb', 2)
        logger.info(f"Increasing memory by {increase_gb}GB")
        await asyncio.sleep(1)


class RecoveryPlaybookRepository:
    """
    Repository of automated recovery playbooks with 5 reactive and 2 proactive playbooks.
    """

    def __init__(self):
        self.playbooks: Dict[str, RecoveryPlaybook] = {}
        self.executor = PlaybookExecutor()
        self.execution_history: List[PlaybookExecution] = []

        # Thread safety
        self._lock = threading.Lock()

        self._initialize_default_playbooks()

    def _initialize_default_playbooks(self):
        """Initialize the 7 default playbooks (5 reactive + 2 proactive)"""

        # 1. Reactive: Memory Leak Recovery
        self.playbooks['memory_leak_recovery'] = RecoveryPlaybook(
            playbook_id='memory_leak_recovery',
            name='Memory Leak Recovery',
            description='Recover from memory leak by restarting services and clearing caches',
            playbook_type=PlaybookType.REACTIVE,
            trigger_conditions=['memory_usage > 90%', 'memory_leak_detected'],
            steps=[
                PlaybookStep(
                    name='alert_team',
                    description='Alert the operations team about memory issue',
                    action='alert_team',
                    parameters={'urgency': 'high', 'message': 'Memory leak detected'}
                ),
                PlaybookStep(
                    name='clear_application_cache',
                    description='Clear application caches to free memory',
                    action='clear_cache',
                    parameters={'cache_type': 'application'}
                ),
                PlaybookStep(
                    name='restart_affected_services',
                    description='Restart services with high memory usage',
                    action='restart_service',
                    parameters={'service_name': 'affected_services'}
                ),
                PlaybookStep(
                    name='run_health_check',
                    description='Verify system health after recovery',
                    action='run_health_check',
                    parameters={'component': 'memory_system'}
                )
            ],
            estimated_duration_seconds=300,
            risk_level='medium',
            tags=['memory', 'leak', 'recovery']
        )

        # 2. Reactive: CPU Overload Recovery
        self.playbooks['cpu_overload_recovery'] = RecoveryPlaybook(
            playbook_id='cpu_overload_recovery',
            name='CPU Overload Recovery',
            description='Recover from CPU overload by scaling resources and restarting services',
            playbook_type=PlaybookType.REACTIVE,
            trigger_conditions=['cpu_usage > 95%', 'cpu_overload_detected'],
            steps=[
                PlaybookStep(
                    name='scale_cpu_resources',
                    description='Scale CPU resources to handle load',
                    action='scale_resources',
                    parameters={'resource_type': 'cpu', 'scale_factor': 2.0}
                ),
                PlaybookStep(
                    name='restart_overloaded_services',
                    description='Restart services causing CPU overload',
                    action='restart_service',
                    parameters={'service_name': 'overloaded_services'}
                ),
                PlaybookStep(
                    name='optimize_performance',
                    description='Run performance optimization routines',
                    action='run_health_check',
                    parameters={'component': 'cpu_performance'}
                )
            ],
            estimated_duration_seconds=240,
            risk_level='low',
            tags=['cpu', 'overload', 'scaling']
        )

        # 3. Reactive: Database Connection Recovery
        self.playbooks['database_connection_recovery'] = RecoveryPlaybook(
            playbook_id='database_connection_recovery',
            name='Database Connection Recovery',
            description='Recover from database connection issues by reconnecting and switching to backup',
            playbook_type=PlaybookType.REACTIVE,
            trigger_conditions=['database_connection_failed', 'query_timeout'],
            steps=[
                PlaybookStep(
                    name='alert_database_team',
                    description='Alert database operations team',
                    action='alert_team',
                    parameters={'urgency': 'critical', 'message': 'Database connection failure'}
                ),
                PlaybookStep(
                    name='reconnect_database',
                    description='Attempt to reconnect to database',
                    action='reconnect_network',
                    parameters={'interface': 'database'}
                ),
                PlaybookStep(
                    name='switch_to_backup_database',
                    description='Switch to backup database if primary fails',
                    action='switch_to_backup',
                    parameters={'system_name': 'backup_database'}
                ),
                PlaybookStep(
                    name='optimize_database_connections',
                    description='Optimize database connection pool',
                    action='optimize_database',
                    parameters={'operation': 'connection_pool'}
                )
            ],
            estimated_duration_seconds=180,
            risk_level='high',
            requires_approval=True,
            tags=['database', 'connection', 'backup']
        )

        # 4. Reactive: Network Latency Recovery
        self.playbooks['network_latency_recovery'] = RecoveryPlaybook(
            playbook_id='network_latency_recovery',
            name='Network Latency Recovery',
            description='Recover from high network latency by switching providers and optimizing connections',
            playbook_type=PlaybookType.REACTIVE,
            trigger_conditions=['network_latency > 500ms', 'packet_loss > 5%'],
            steps=[
                PlaybookStep(
                    name='reconnect_network_interfaces',
                    description='Reconnect network interfaces',
                    action='reconnect_network',
                    parameters={'interface': 'primary'}
                ),
                PlaybookStep(
                    name='switch_network_provider',
                    description='Switch to backup network provider',
                    action='switch_to_backup',
                    parameters={'system_name': 'backup_network'}
                ),
                PlaybookStep(
                    name='optimize_network_settings',
                    description='Optimize network settings for low latency',
                    action='run_health_check',
                    parameters={'component': 'network_optimization'}
                )
            ],
            estimated_duration_seconds=120,
            risk_level='medium',
            tags=['network', 'latency', 'optimization']
        )

        # 5. Reactive: Trading Halt Recovery
        self.playbooks['trading_halt_recovery'] = RecoveryPlaybook(
            playbook_id='trading_halt_recovery',
            name='Trading Halt Recovery',
            description='Recover from trading halt by restarting trading services and validating connectivity',
            playbook_type=PlaybookType.REACTIVE,
            trigger_conditions=['trading_halted', 'order_rejection_rate > 50%'],
            steps=[
                PlaybookStep(
                    name='emergency_alert',
                    description='Send emergency alert to trading team',
                    action='alert_team',
                    parameters={'urgency': 'critical', 'message': 'Trading system halted'}
                ),
                PlaybookStep(
                    name='restart_trading_engine',
                    description='Restart the trading engine service',
                    action='restart_service',
                    parameters={'service_name': 'trading_engine'}
                ),
                PlaybookStep(
                    name='validate_exchange_connectivity',
                    description='Validate connectivity to exchanges',
                    action='run_health_check',
                    parameters={'component': 'exchange_connectivity'}
                ),
                PlaybookStep(
                    name='rollback_to_last_stable_version',
                    description='Rollback to last stable version if needed',
                    action='rollback_deployment',
                    parameters={'version': 'last_stable'}
                )
            ],
            estimated_duration_seconds=180,
            risk_level='high',
            requires_approval=True,
            tags=['trading', 'halt', 'emergency']
        )

        # 6. Proactive: Memory Usage Prevention
        self.playbooks['memory_usage_prevention'] = RecoveryPlaybook(
            playbook_id='memory_usage_prevention',
            name='Memory Usage Prevention',
            description='Proactive playbook to prevent memory issues during predicted high usage periods',
            playbook_type=PlaybookType.PROACTIVE,
            trigger_conditions=['predicted_memory_usage > 80%', 'memory_trend_increasing'],
            steps=[
                PlaybookStep(
                    name='increase_memory_allocation',
                    description='Proactively increase memory allocation',
                    action='increase_memory',
                    parameters={'increase_gb': 4}
                ),
                PlaybookStep(
                    name='schedule_cache_clearing',
                    description='Schedule cache clearing during low activity',
                    action='clear_cache',
                    parameters={'cache_type': 'scheduled'}
                ),
                PlaybookStep(
                    name='enable_memory_monitoring',
                    description='Enable enhanced memory monitoring',
                    action='run_health_check',
                    parameters={'component': 'memory_monitoring'}
                )
            ],
            estimated_duration_seconds=60,
            risk_level='low',
            tags=['memory', 'prevention', 'proactive']
        )

        # 7. Proactive: Performance Optimization
        self.playbooks['performance_optimization'] = RecoveryPlaybook(
            playbook_id='performance_optimization',
            name='Performance Optimization',
            description='Proactive playbook to optimize system performance during predicted load increase',
            playbook_type=PlaybookType.PROACTIVE,
            trigger_conditions=['predicted_load_increase', 'cpu_usage_trend > 70%'],
            steps=[
                PlaybookStep(
                    name='scale_resources_proactively',
                    description='Scale resources before load increase',
                    action='scale_resources',
                    parameters={'resource_type': 'cpu', 'scale_factor': 1.5}
                ),
                PlaybookStep(
                    name='optimize_database_performance',
                    description='Run database performance optimization',
                    action='optimize_database',
                    parameters={'operation': 'performance_tuning'}
                ),
                PlaybookStep(
                    name='clear_performance_impacting_cache',
                    description='Clear caches that may impact performance',
                    action='clear_cache',
                    parameters={'cache_type': 'performance_critical'}
                ),
                PlaybookStep(
                    name='enable_performance_monitoring',
                    description='Enable enhanced performance monitoring',
                    action='run_health_check',
                    parameters={'component': 'performance_metrics'}
                )
            ],
            estimated_duration_seconds=300,
            risk_level='low',
            tags=['performance', 'optimization', 'proactive']
        )

    def get_playbook(self, playbook_id: str) -> Optional[RecoveryPlaybook]:
        """Get a playbook by ID"""
        return self.playbooks.get(playbook_id)

    def list_playbooks(self, playbook_type: Optional[PlaybookType] = None,
                      tags: Optional[List[str]] = None) -> List[RecoveryPlaybook]:
        """List available playbooks with optional filtering"""
        playbooks = list(self.playbooks.values())

        if playbook_type:
            playbooks = [p for p in playbooks if p.playbook_type == playbook_type]

        if tags:
            playbooks = [p for p in playbooks if any(tag in p.tags for tag in tags)]

        return playbooks

    def find_matching_playbooks(self, conditions: List[str]) -> List[RecoveryPlaybook]:
        """Find playbooks that match given trigger conditions"""
        matching = []

        for playbook in self.playbooks.values():
            if any(condition in playbook.trigger_conditions for condition in conditions):
                matching.append(playbook)

        return matching

    async def execute_playbook(self, playbook_id: str,
                             context: Dict[str, Any] = None) -> Optional[PlaybookExecution]:
        """Execute a playbook by ID"""
        playbook = self.get_playbook(playbook_id)
        if not playbook:
            logger.error(f"Playbook not found: {playbook_id}")
            return None

        execution = await self.executor.execute_playbook(playbook, context)

        with self._lock:
            self.execution_history.append(execution)

        return execution

    def get_execution_history(self, playbook_id: Optional[str] = None,
                            status: Optional[PlaybookStatus] = None) -> List[PlaybookExecution]:
        """Get execution history with optional filtering"""
        history = self.execution_history.copy()

        if playbook_id:
            history = [exec for exec in history if exec.playbook_id == playbook_id]

        if status:
            history = [exec for exec in history if exec.status == status]

        return history

    def get_playbook_statistics(self) -> Dict[str, Any]:
        """Get statistics about playbook usage and success rates"""
        total_executions = len(self.execution_history)
        if total_executions == 0:
            return {'total_executions': 0}

        successful = sum(1 for exec in self.execution_history
                        if exec.status == PlaybookStatus.COMPLETED)
        failed = sum(1 for exec in self.execution_history
                    if exec.status == PlaybookStatus.FAILED)

        success_rate = successful / total_executions * 100

        # Playbook-specific stats
        playbook_stats = {}
        for playbook_id in self.playbooks.keys():
            executions = [exec for exec in self.execution_history
                         if exec.playbook_id == playbook_id]
            if executions:
                success_count = sum(1 for exec in executions
                                  if exec.status == PlaybookStatus.COMPLETED)
                playbook_stats[playbook_id] = {
                    'total_executions': len(executions),
                    'success_rate': success_count / len(executions) * 100
                }

        return {
            'total_executions': total_executions,
            'successful_executions': successful,
            'failed_executions': failed,
            'success_rate': success_rate,
            'playbook_stats': playbook_stats
        }

    def save_repository_state(self, path: str):
        """Save repository state and execution history"""
        state = {
            'execution_history': [
                {
                    'execution_id': exec.execution_id,
                    'playbook_id': exec.playbook_id,
                    'start_time': exec.start_time.isoformat(),
                    'end_time': exec.end_time.isoformat() if exec.end_time else None,
                    'status': exec.status.value,
                    'results': exec.results,
                    'errors': exec.errors
                }
                for exec in self.execution_history
            ],
            'playbook_statistics': self.get_playbook_statistics(),
            'timestamp': datetime.now().isoformat()
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Saved repository state to {path}")

    def load_repository_state(self, path: str):
        """Load repository state and execution history"""
        try:
            with open(path, 'r') as f:
                state = json.load(f)

            # Reconstruct execution history
            self.execution_history = []
            for exec_data in state.get('execution_history', []):
                execution = PlaybookExecution(
                    execution_id=exec_data['execution_id'],
                    playbook_id=exec_data['playbook_id'],
                    start_time=datetime.fromisoformat(exec_data['start_time']),
                    end_time=datetime.fromisoformat(exec_data['end_time']) if exec_data['end_time'] else None,
                    status=PlaybookStatus(exec_data['status']),
                    results=exec_data.get('results', {}),
                    errors=exec_data.get('errors', [])
                )
                self.execution_history.append(execution)

            logger.info(f"Loaded repository state from {path}")

        except Exception as e:
            logger.error(f"Failed to load repository state: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def test_recovery_playbook_repository():
        # Create repository
        repo = RecoveryPlaybookRepository()

        # List all playbooks
        print("Available Playbooks:")
        for playbook in repo.list_playbooks():
            print(f"  - {playbook.name} ({playbook.playbook_type.value})")

        # List reactive playbooks
        print("\nReactive Playbooks:")
        for playbook in repo.list_playbooks(PlaybookType.REACTIVE):
            print(f"  - {playbook.name}")

        # List proactive playbooks
        print("\nProactive Playbooks:")
        for playbook in repo.list_playbooks(PlaybookType.PROACTIVE):
            print(f"  - {playbook.name}")

        # Execute a playbook
        execution = await repo.execute_playbook('memory_leak_recovery', {'test': True})

        if execution:
            print(f"\nExecuted playbook: {execution.playbook_id}")
            print(f"Status: {execution.status.value}")
            print(f"Results: {execution.results}")

        # Get statistics
        stats = repo.get_playbook_statistics()
        print(f"\nPlaybook Statistics: {stats}")

        # Save state
        repo.save_repository_state("playbook_repository_state.json")

    # Run test
    asyncio.run(test_recovery_playbook_repository())