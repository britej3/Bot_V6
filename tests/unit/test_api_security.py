"""
Unit Tests for API Security Components
====================================
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from src.api.models import (
    APIKey,
    SecurityEvent,
    RateLimit,
    AuditLog,
    TradingAuditLog,
    SystemHealthCheck,
    ComplianceReport
)


class TestAPIKey:
    """Test cases for APIKey model"""

    def test_api_key_generation(self):
        """Test API key generation"""
        raw_key, api_key = APIKey.generate_key("test_key", ["read", "trade"])

        assert len(raw_key) > 32  # Should be a secure random string
        assert api_key.key_id.startswith("key_")
        assert api_key.name == "test_key"
        assert "read" in api_key.permissions
        assert "trade" in api_key.permissions
        assert not api_key.is_expired()

    def test_api_key_verification(self):
        """Test API key verification"""
        raw_key, api_key = APIKey.generate_key("test_key", ["read"])

        # Should verify correctly
        assert api_key.verify_key(raw_key)

        # Should reject invalid key
        assert not api_key.verify_key("invalid_key")

        # Should reject tampered key
        tampered_key = raw_key[:-5] + "xxxxx"
        assert not api_key.verify_key(tampered_key)

    def test_api_key_expiration(self):
        """Test API key expiration"""
        raw_key, api_key = APIKey.generate_key("test_key", ["read"])

        # Set expiration to past time
        api_key.expires_at = time.time() - 3600  # 1 hour ago

        assert api_key.is_expired()

        # Set expiration to future time
        api_key.expires_at = time.time() + 3600  # 1 hour from now

        assert not api_key.is_expired()

    def test_api_key_without_expiration(self):
        """Test API key without expiration date"""
        raw_key, api_key = APIKey.generate_key("test_key", ["read"])

        # No expiration set
        api_key.expires_at = None

        assert not api_key.is_expired()


class TestSecurityEvent:
    """Test cases for SecurityEvent model"""

    def test_security_event_creation(self):
        """Test security event creation"""
        event = SecurityEvent(
            event_type="auth_failure",
            user_id="user123",
            ip_address="192.168.1.100",
            endpoint="/api/trade",
            details={"attempt_count": 3}
        )

        assert event.event_id.startswith("sec_")
        assert event.event_type == "auth_failure"
        assert event.user_id == "user123"
        assert event.ip_address == "192.168.1.100"
        assert event.endpoint == "/api/trade"
        assert event.details["attempt_count"] == 3
        assert event.severity == "medium"
        assert isinstance(event.timestamp, float)


class TestRateLimit:
    """Test cases for RateLimit model"""

    def test_rate_limit_creation(self):
        """Test rate limit creation"""
        rate_limit = RateLimit(
            identifier="192.168.1.100",
            limit_type="requests_per_minute",
            limit=60,
            window_size=60
        )

        assert rate_limit.identifier == "192.168.1.100"
        assert rate_limit.limit == 60
        assert rate_limit.current_count == 0
        assert not rate_limit.is_blocked()

    def test_rate_limit_increment(self):
        """Test rate limit increment"""
        rate_limit = RateLimit(
            identifier="192.168.1.100",
            limit_type="requests_per_minute",
            limit=5,
            window_size=60
        )

        # Should allow requests until limit is reached
        for i in range(5):
            assert rate_limit.increment()
            assert rate_limit.current_count == i + 1

        # Should block after limit is reached
        assert not rate_limit.increment()
        assert rate_limit.is_blocked()

    def test_rate_limit_window_reset(self):
        """Test rate limit window reset"""
        rate_limit = RateLimit(
            identifier="192.168.1.100",
            limit_type="requests_per_minute",
            limit=5,
            window_size=1  # 1 second window for testing
        )

        # Fill up the limit
        for _ in range(5):
            rate_limit.increment()

        assert rate_limit.is_blocked()

        # Wait for window to reset
        time.sleep(1.1)

        # Should allow requests again
        assert rate_limit.increment()
        assert not rate_limit.is_blocked()

    def test_rate_limit_not_blocked_initially(self):
        """Test that rate limit is not blocked initially"""
        rate_limit = RateLimit(
            identifier="192.168.1.100",
            limit_type="requests_per_minute",
            limit=60,
            window_size=60
        )

        assert not rate_limit.is_blocked()
        assert rate_limit.increment()


class TestAuditLog:
    """Test cases for AuditLog model"""

    def test_audit_log_creation(self):
        """Test audit log creation"""
        audit_log = AuditLog(
            user_id="user123",
            action="CREATE_ORDER",
            resource="orders",
            method="POST",
            status_code=201,
            ip_address="192.168.1.100",
            user_agent="Mozilla/5.0",
            request_id="req_123",
            response_time=0.045
        )

        assert audit_log.log_id.startswith("audit_")
        assert audit_log.user_id == "user123"
        assert audit_log.action == "CREATE_ORDER"
        assert audit_log.resource == "orders"
        assert audit_log.method == "POST"
        assert audit_log.status_code == 201
        assert audit_log.ip_address == "192.168.1.100"
        assert audit_log.user_agent == "Mozilla/5.0"
        assert audit_log.request_id == "req_123"
        assert audit_log.response_time == 0.045
        assert isinstance(audit_log.timestamp, float)


class TestTradingAuditLog:
    """Test cases for TradingAuditLog model"""

    def test_trading_audit_log_creation(self):
        """Test trading audit log creation"""
        trading_audit = TradingAuditLog(
            user_id="trader123",
            action="EXECUTE_TRADE",
            resource="orders",
            method="POST",
            status_code=200,
            symbol="BTC/USDT",
            side="buy",
            quantity=0.1,
            price=50000.0,
            order_type="market",
            strategy_name="scalping",
            risk_score=0.85,
            compliance_flags=["high_frequency_trading"]
        )

        assert trading_audit.symbol == "BTC/USDT"
        assert trading_audit.side == "buy"
        assert trading_audit.quantity == 0.1
        assert trading_audit.price == 50000.0
        assert trading_audit.order_type == "market"
        assert trading_audit.strategy_name == "scalping"
        assert trading_audit.risk_score == 0.85
        assert "high_frequency_trading" in trading_audit.compliance_flags


class TestSystemHealthCheck:
    """Test cases for SystemHealthCheck model"""

    def test_system_health_check_creation(self):
        """Test system health check creation"""
        health_check = SystemHealthCheck(
            component="trading_engine",
            status="healthy",
            response_time=0.023,
            details={"active_connections": 150}
        )

        assert health_check.check_id.startswith("health_")
        assert health_check.component == "trading_engine"
        assert health_check.status == "healthy"
        assert health_check.response_time == 0.023
        assert health_check.details["active_connections"] == 150
        assert isinstance(health_check.timestamp, float)

    def test_system_health_check_unhealthy(self):
        """Test system health check with unhealthy status"""
        health_check = SystemHealthCheck(
            component="database",
            status="unhealthy",
            error_message="Connection timeout",
            details={"connection_pool_exhausted": True}
        )

        assert health_check.status == "unhealthy"
        assert health_check.error_message == "Connection timeout"
        assert health_check.details["connection_pool_exhausted"] is True


class TestComplianceReport:
    """Test cases for ComplianceReport model"""

    def test_compliance_report_creation(self):
        """Test compliance report creation"""
        report = ComplianceReport(
            report_type="daily_trading_summary",
            period_start=time.time() - 86400,  # 24 hours ago
            period_end=time.time(),
            total_trades=150,
            compliance_violations=2,
            risk_exposures=[
                {"asset": "BTC/USDT", "exposure": 0.85, "limit": 0.8},
                {"asset": "ETH/USDT", "exposure": 0.92, "limit": 0.9}
            ],
            recommendations=[
                "Reduce BTC/USDT position by 5%",
                "Review ETH/USDT risk parameters"
            ]
        )

        assert report.report_id.startswith("comp_")
        assert report.report_type == "daily_trading_summary"
        assert report.total_trades == 150
        assert report.compliance_violations == 2
        assert len(report.risk_exposures) == 2
        assert len(report.recommendations) == 2
        assert report.status == "draft"

    def test_compliance_report_status_update(self):
        """Test compliance report status updates"""
        report = ComplianceReport(
            report_type="weekly_risk_assessment",
            period_start=time.time() - 604800,  # 1 week ago
            period_end=time.time()
        )

        assert report.status == "draft"

        report.status = "final"
        assert report.status == "final"

        report.status = "approved"
        assert report.status == "approved"


# Integration tests
class TestSecurityIntegration:
    """Integration tests for security components"""

    def test_api_key_lifecycle(self):
        """Test complete API key lifecycle"""
        # Generate key
        raw_key, api_key = APIKey.generate_key("integration_test", ["read", "trade"])

        # Verify it works
        assert api_key.verify_key(raw_key)
        assert not api_key.is_expired()

        # Mark as used
        api_key.last_used_at = time.time()

        # Simulate expiration
        api_key.expires_at = time.time() - 3600
        assert api_key.is_expired()

        # Deactivate
        api_key.is_active = False

        # Should still verify but be inactive
        assert api_key.verify_key(raw_key)
        assert not api_key.is_active

    def test_rate_limiting_scenario(self):
        """Test realistic rate limiting scenario"""
        rate_limit = RateLimit(
            identifier="api_client_123",
            limit_type="requests_per_minute",
            limit=30,
            window_size=60
        )

        # Simulate normal usage
        for _ in range(30):
            assert rate_limit.increment()

        # Should be blocked now
        assert not rate_limit.increment()
        assert rate_limit.is_blocked()

        # Simulate waiting for reset (in test, we manually reset window)
        rate_limit.window_start = time.time() - 61  # Move window back

        # Should allow requests again
        assert rate_limit.increment()
        assert not rate_limit.is_blocked()

    def test_audit_trail_completeness(self):
        """Test that audit trail captures all necessary information"""
        audit_entries = []

        # Simulate a trading session
        actions = [
            ("user_login", "POST", 200),
            ("api_key_generate", "POST", 201),
            ("order_create", "POST", 201),
            ("order_execute", "PUT", 200),
            ("position_query", "GET", 200),
            ("user_logout", "POST", 200)
        ]

        for action, method, status_code in actions:
            audit_log = AuditLog(
                user_id="test_user",
                action=action.upper(),
                resource="trading_system",
                method=method,
                status_code=status_code,
                ip_address="192.168.1.100",
                request_id=f"req_{len(audit_entries) + 1}"
            )
            audit_entries.append(audit_log)

        # Verify all actions are captured
        assert len(audit_entries) == 6
        assert all(entry.user_id == "test_user" for entry in audit_entries)
        assert all(entry.ip_address == "192.168.1.100" for entry in audit_entries)

        # Verify timestamps are sequential
        timestamps = [entry.timestamp for entry in audit_entries]
        assert timestamps == sorted(timestamps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])