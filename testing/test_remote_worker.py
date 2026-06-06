from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import execnet
import pytest

from xdist.remote import RemoteWorker


class TestRemoteWorkerSendCommand:
    """Tests for RemoteWorker.send_command() method."""

    def test_send_command_success(self) -> None:
        """Test normal successful command sending."""
        mock_channel = MagicMock(spec=execnet.Channel)
        mock_channel.send.return_value = None

        worker = RemoteWorker(channel=mock_channel, timeout=5.0, max_retries=3)
        result = worker.send_command("runtests", indices=[0, 1, 2])

        assert result is True
        mock_channel.send.assert_called_once_with(
            ("runtests", {"indices": [0, 1, 2]}), timeout=5.0
        )

    def test_send_command_with_empty_kwargs(self) -> None:
        """Test sending command with no additional arguments."""
        mock_channel = MagicMock(spec=execnet.Channel)

        worker = RemoteWorker(channel=mock_channel)
        result = worker.send_command("shutdown")

        assert result is True
        mock_channel.send.assert_called_once_with(
            ("shutdown", {}), timeout=5.0
        )

    def test_send_command_timeout_then_success(self) -> None:
        """Test command succeeds after initial timeout retries."""
        mock_channel = MagicMock(spec=execnet.Channel)
        timeout_error = execnet.Channel.TimeoutError("timeout")

        call_count = 0

        def side_effect(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise timeout_error

        mock_channel.send.side_effect = side_effect

        worker = RemoteWorker(channel=mock_channel, timeout=2.0, max_retries=3)
        result = worker.send_command("runtests_all")

        assert result is True
        assert call_count == 3
        assert mock_channel.send.call_count == 3

    def test_send_command_timeout_exhausts_retries(self) -> None:
        """Test TimeoutError raised when all retries are exhausted due to timeouts."""
        mock_channel = MagicMock(spec=execnet.Channel)
        timeout_error = execnet.Channel.TimeoutError("timeout")
        mock_channel.send.side_effect = timeout_error

        worker = RemoteWorker(channel=mock_channel, timeout=1.0, max_retries=3)

        with pytest.raises(TimeoutError) as exc_info:
            worker.send_command("runtests", indices=[0])

        assert "Failed to send command 'runtests' after 3 attempts" in str(exc_info.value)
        assert mock_channel.send.call_count == 3

    def test_send_command_timeout_retry_delays(self) -> None:
        """Test that retry delays increase with each attempt."""
        mock_channel = MagicMock(spec=execnet.Channel)
        timeout_error = execnet.Channel.TimeoutError("timeout")

        call_times: list[float] = []

        def side_effect(*args: Any, **kwargs: Any) -> None:
            call_times.append(time.monotonic())
            raise timeout_error

        mock_channel.send.side_effect = side_effect

        worker = RemoteWorker(channel=mock_channel, timeout=0.1, max_retries=3)

        with pytest.raises(TimeoutError):
            worker.send_command("test_cmd")

        assert len(call_times) == 3
        delay_1 = call_times[1] - call_times[0]
        delay_2 = call_times[2] - call_times[1]
        assert delay_2 > delay_1

    def test_send_command_connection_error_immediate(self) -> None:
        """Test ConnectionError raised immediately on connection break."""
        mock_channel = MagicMock(spec=execnet.Channel)
        remote_error = execnet.RemoteError("connection broken")
        mock_channel.send.side_effect = remote_error

        worker = RemoteWorker(channel=mock_channel, timeout=5.0, max_retries=3)

        with pytest.raises(ConnectionError) as exc_info:
            worker.send_command("runtests", indices=[0])

        assert "Connection broken while sending command 'runtests'" in str(exc_info.value)
        mock_channel.send.assert_called_once()

    def test_send_command_connection_error_no_retry(self) -> None:
        """Test that connection errors are not retried."""
        mock_channel = MagicMock(spec=execnet.Channel)
        remote_error = execnet.RemoteError("channel closed")
        mock_channel.send.side_effect = remote_error

        worker = RemoteWorker(channel=mock_channel, max_retries=5)

        with pytest.raises(ConnectionError):
            worker.send_command("shutdown")

        assert mock_channel.send.call_count == 1

    def test_send_command_default_parameters(self) -> None:
        """Test that default timeout and max_retries are used."""
        mock_channel = MagicMock(spec=execnet.Channel)

        worker = RemoteWorker(channel=mock_channel)

        assert worker.timeout == 5.0
        assert worker.max_retries == 3

    def test_send_command_custom_parameters(self) -> None:
        """Test custom timeout and max_retries configuration."""
        mock_channel = MagicMock(spec=execnet.Channel)

        worker = RemoteWorker(
            channel=mock_channel, timeout=10.0, max_retries=5
        )

        assert worker.timeout == 10.0
        assert worker.max_retries == 5
        worker.send_command("test")
        mock_channel.send.assert_called_once_with(
            ("test", {}), timeout=10.0
        )

    def test_send_command_preserves_exception_chain(self) -> None:
        """Test that the original exception is preserved in __cause__."""
        mock_channel = MagicMock(spec=execnet.Channel)
        timeout_error = execnet.Channel.TimeoutError("original timeout")
        mock_channel.send.side_effect = timeout_error

        worker = RemoteWorker(channel=mock_channel, timeout=1.0, max_retries=2)

        with pytest.raises(TimeoutError) as exc_info:
            worker.send_command("test")

        assert exc_info.value.__cause__ is timeout_error

    def test_send_command_connection_error_preserves_exception_chain(self) -> None:
        """Test that RemoteError is preserved in ConnectionError __cause__."""
        mock_channel = MagicMock(spec=execnet.Channel)
        remote_error = execnet.RemoteError("connection lost")
        mock_channel.send.side_effect = remote_error

        worker = RemoteWorker(channel=mock_channel)

        with pytest.raises(ConnectionError) as exc_info:
            worker.send_command("test")

        assert exc_info.value.__cause__ is remote_error
