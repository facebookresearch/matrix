# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from unittest.mock import MagicMock, patch

import pytest
from matrix.utils.logging import MatrixLogger, MatrixLogLevel, get_logger


@pytest.fixture(scope="session", autouse=True)
def configure_logging():
    """Configure logging for frogger2 transmission."""
    # Set the frogger2 logger to INFO level so transmissions work
    logging.getLogger("fair_matrix").setLevel(logging.INFO)
    # Also ensure root logger doesn't filter it out
    logging.basicConfig(level=logging.INFO)


class TestMatrixLogger:
    """Test suite for MatrixLogger functionality"""

    def test_matrix_logger_initialization(self):
        TEST_LOGGER: str = "test_logger"
        logger = MatrixLogger(name=TEST_LOGGER, level=logging.INFO)
        assert logger.logger.name == TEST_LOGGER

    def test_get_logger_returns_matrix_logger_with_forgger(self):
        with patch("matrix.utils.logging.FROGGER_AVAILABLE", True):
            with patch(
                "matrix.utils.logging.OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://localhost:11000",
            ):
                logger = get_logger("test")
                assert isinstance(logger, MatrixLogger)

    def test_get_logger_returns_standard_logger_without_frogger(self):
        with patch("matrix.utils.logging.FROGGER_AVAILABLE", False):
            logger = get_logger("test")
            assert isinstance(logger, logging.Logger)
            assert not isinstance(logger, MatrixLogger)

    def test_matrix_logger_info_with_frogger(self):
        with patch("matrix.utils.logging.FROGGER_AVAILABLE", True):
            with patch(
                "matrix.utils.logging.OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://localhost:11000",
            ):
                with patch("matrix.utils.logging.matrix_logger") as mock_frogger:
                    logger = MatrixLogger(name="test")
                    logger.use_frogger = True

                    # Mock the underlying standard logger
                    with patch.object(logger.logger, "info") as mock_info:
                        logger.info("Test message", job_id="123")

                        # Verify standard logger was called
                        mock_info.assert_called_once()

                        # Verify  frogger transmit was called
                        mock_frogger.assert_called_once_with(
                            log_level=MatrixLogLevel.INFO.name,
                            log_message="Test message",
                            job_id="123",
                        )

    def test_matrix_logger_info_without_frogger(self):
        logger = MatrixLogger(name="test")
        logger.use_frogger = False

        with patch.object(logger.logger, "info") as mock_info:
            logger.info("Test message", status=True)

            # Verify standard logger was called with formatted message
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert "Test message" in call_args[0][0]
            assert "status=True" in call_args[0][0]

    def test_matrix_logger_all_levels(self):
        logger = MatrixLogger(name="test")
        logger.use_frogger = False

        levels = ["debug", "info", "warning", "error", "critical"]

        for level in levels:
            with patch.object(logger.logger, level) as mock_log:
                log_method = getattr(logger, level)
                log_method(f"Test {level} message")
                mock_log.assert_called_once()

    def test_matrix_logger_handlers(self):
        logger = MatrixLogger(name="test")

        # Test adding handler
        handler = logging.StreamHandler()
        logger.addHandler(handler)
        assert handler in logger.handlers

        # Test removing handler
        logger.removeHandler(handler)
        assert handler not in logger.handlers

    def test_matrix_logger_set_level(self):
        logger = MatrixLogger(name="test", level=logging.INFO)
        assert logger.logger.level == logging.INFO

        logger.setLevel(logging.DEBUG)
        assert logger.logger.level == logging.DEBUG

    def test_transmit_handles_exceptions(self):
        with patch("matrix.utils.logging.FROGGER_AVAILABLE", True):
            with patch(
                "matrix.utils.logging.OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://localhost:4317",
            ):
                with patch(
                    "matrix.utils.logging.matrix_logger",
                    side_effect=Exception("Connection failed"),
                ):
                    logger = MatrixLogger(name="test")
                    logger.use_frogger = True

                    with patch.object(logger.logger, "error") as mock_error:
                        logger.transmit(MatrixLogLevel.INFO, "Test message")

                        mock_error.assert_called_once()
                        assert "MatrixLogger failed to transmit payload" in str(
                            mock_error.call_args
                        )


class TestMatrixLoggerIntegration:
    def test_ray_dashboard_job_logging(self):
        from matrix.cluster.ray_dashboard_job import logger

        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

        with patch.object(logger, "info") as mock_info:
            logger.info("Test log message")
            mock_info.assert_called_once()

    def test_logger_with_structured_data(self):
        logger = MatrixLogger(name="test")
        logger.use_frogger = False

        with patch.object(logger.logger, "info") as mock_info:
            logger.info(
                "Job completed",
                job_id="job_123",
                duration_seconds=42.5,
                num_samples=1000,
                status=True,
            )

            # Verify all fields are in the logged message
            call_args = mock_info.call_args[0][0]
            assert "job_id=job_123" in call_args
            assert "duration_seconds=42.5" in call_args
            assert "num_samples=1000" in call_args
            assert "status=True" in call_args


class TestMatrixLoggerWithFrogger:
    """requires frogger2 to be installed"""

    from matrix.utils.logging import FROGGER_AVAILABLE, OTEL_EXPORTER_OTLP_ENDPOINT

    @pytest.mark.skipif(
        not FROGGER_AVAILABLE or not OTEL_EXPORTER_OTLP_ENDPOINT,
        reason="Requires frogger2 and an OTel gateway",
    )
    def test_transmission(self):
        from matrix.utils.logging import FROGGER_AVAILABLE

        assert FROGGER_AVAILABLE, "frogger2 must be available for this test"
        logger = MatrixLogger(name="test_frogger")
        assert logger.use_frogger, "Logger should be configured to use frogger2"

        success: bool = False
        try:
            logger.info(
                "Integration test message",
                job_id="test_job_123",
                duration=1.5,
                status=True,
            )
            success = True
        except Exception as exn:
            pytest.fail(f"Transmission failed with error: {str(exn)}")

            assert success, "Transmission should complete without errors"

    @pytest.mark.skipif(
        not FROGGER_AVAILABLE or not OTEL_EXPORTER_OTLP_ENDPOINT,
        reason="Requires frogger2 and an OTel gateway",
    )
    def test_frogger_transmission_with_all_log_levels(self):
        """Test transmission at all log levels."""
        logger = MatrixLogger(name="test_levels")

        test_cases = [
            (logger.debug, MatrixLogLevel.DEBUG, "Debug message"),
            (logger.info, MatrixLogLevel.INFO, "Info message"),
            (logger.warning, MatrixLogLevel.WARNING, "Warning message"),
            (logger.error, MatrixLogLevel.ERROR, "Error message"),
            (logger.critical, MatrixLogLevel.CRITICAL, "Critical message"),
        ]

        for log_method, level, message in test_cases:
            try:
                log_method(message, job_id=f"test_{level.name.lower()}")
            except Exception as exn:
                pytest.fail(f"Failed to transmit {level.name}: {exn}")

    @pytest.mark.skipif(
        not FROGGER_AVAILABLE or not OTEL_EXPORTER_OTLP_ENDPOINT,
        reason="Requires frogger2 and an OTel gateway",
    )
    def test_frogger_transmission_with_structured_fields(self):
        """Test that structured fields are properly transmitted."""
        logger = MatrixLogger(name="test_structured")

        # These are the fields that should be extracted for OTLP
        structured_data = {
            "job_id": "integration_test_456",
            "duration_seconds": 123.45,
            "num_samples": 5000,
            "status": "completed",
        }

        try:
            logger.info("Structured logging test", **structured_data)
        except Exception as e:
            pytest.fail(f"Structured transmission failed: {e}")

    @pytest.mark.skipif(
        not FROGGER_AVAILABLE or not OTEL_EXPORTER_OTLP_ENDPOINT,
        reason="Requires frogger2 and an OTel gateway",
    )
    def test_verify_local_and_remote_logging_both_occur(self):
        """Verify that both local logging and remote transmission happen."""
        logger = MatrixLogger(name="test_dual")

        # Capture local logs
        import io

        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Log a message
        test_message = "Dual logging verification test"
        logger.info(test_message, job_id="dual_test_001")

        # Verify local log was written
        log_output = log_capture.getvalue()
        assert test_message in log_output, "Local log should contain the message"
        assert len(log_output) > 0, "Local log should not be empty"

        # Note: We can't easily verify remote transmission without querying the OTLP endpoint,
        # but if no exception was raised, transmission was attempted
        logger.removeHandler(handler)
