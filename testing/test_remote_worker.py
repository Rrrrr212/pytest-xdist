from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xdist.remote import RemoteWorker


class TestRemoteWorkerSendCommand:
    def test_send_command_normal(self) -> None:
        mock_channel = MagicMock()
        worker = RemoteWorker(mock_channel)

        worker.send_command("runtests", indices=[1, 2, 3])

        mock_channel.send.assert_called_once_with(
            ("runtests", {"indices": [1, 2, 3]})
        )

    def test_send_command_with_named_params(self) -> None:
        mock_channel = MagicMock()
        worker = RemoteWorker(mock_channel)

        worker.send_command("steal", indices=[0, 1], reason="test")

        mock_channel.send.assert_called_once_with(
            ("steal", {"indices": [0, 1], "reason": "test"})
        )

    def test_send_command_timeout_retry_succeeds(self) -> None:
        mock_channel = MagicMock()
        mock_channel.TimeoutError = TimeoutError
        mock_channel.send.side_effect = [TimeoutError("timeout"), None]

        worker = RemoteWorker(mock_channel, retry_count=3)

        worker.send_command("runtests", indices=[1])

        assert mock_channel.send.call_count == 2

    def test_send_command_timeout_exhausted(self) -> None:
        mock_channel = MagicMock()
        mock_channel.TimeoutError = TimeoutError
        mock_channel.send.side_effect = TimeoutError("timeout")

        worker = RemoteWorker(mock_channel, retry_count=3)

        with pytest.raises(TimeoutError, match="timed out after 3 attempts"):
            worker.send_command("runtests", indices=[1])

        assert mock_channel.send.call_count == 3

    def test_send_command_timeout_exhausted_custom_retry(self) -> None:
        mock_channel = MagicMock()
        mock_channel.TimeoutError = TimeoutError
        mock_channel.send.side_effect = TimeoutError("timeout")

        worker = RemoteWorker(mock_channel, retry_count=5)

        with pytest.raises(TimeoutError, match="timed out after 5 attempts"):
            worker.send_command("shutdown")

        assert mock_channel.send.call_count == 5

    def test_send_command_connection_disconnect_eof(self) -> None:
        mock_channel = MagicMock()
        mock_channel.send.side_effect = EOFError("connection closed")

        worker = RemoteWorker(mock_channel)

        with pytest.raises(ConnectionError, match="Connection lost"):
            worker.send_command("shutdown")

        mock_channel.send.assert_called_once()

    def test_send_command_connection_disconnect_oserror(self) -> None:
        mock_channel = MagicMock()
        mock_channel.send.side_effect = OSError("broken pipe")

        worker = RemoteWorker(mock_channel)

        with pytest.raises(ConnectionError, match="Connection lost"):
            worker.send_command("steal", indices=[0])

        mock_channel.send.assert_called_once()

    def test_send_command_connection_disconnect_on_first_attempt(self) -> None:
        mock_channel = MagicMock()
        mock_channel.send.side_effect = OSError("connection reset")

        worker = RemoteWorker(mock_channel, retry_count=3)

        with pytest.raises(ConnectionError, match="Connection lost"):
            worker.send_command("runtests_all")

        mock_channel.send.assert_called_once()

    def test_send_command_connection_disconnect_after_timeout(self) -> None:
        mock_channel = MagicMock()
        mock_channel.TimeoutError = TimeoutError
        mock_channel.send.side_effect = [
            TimeoutError("timeout"),
            OSError("connection lost"),
        ]

        worker = RemoteWorker(mock_channel, retry_count=3)

        with pytest.raises(ConnectionError, match="Connection lost"):
            worker.send_command("runtests", indices=[1])

        assert mock_channel.send.call_count == 2

    def test_send_command_multiple_commands(self) -> None:
        mock_channel = MagicMock()
        worker = RemoteWorker(mock_channel)

        worker.send_command("runtests", indices=[1])
        worker.send_command("runtests", indices=[2])
        worker.send_command("shutdown")

        assert mock_channel.send.call_count == 3
        assert mock_channel.send.call_args_list == [
            (("runtests", {"indices": [1]}),),
            (("runtests", {"indices": [2]}),),
            (("shutdown", {}),),
        ]

    def test_send_command_default_retry_count(self) -> None:
        mock_channel = MagicMock()
        mock_channel.TimeoutError = TimeoutError
        mock_channel.send.side_effect = TimeoutError("timeout")

        worker = RemoteWorker(mock_channel)

        with pytest.raises(TimeoutError):
            worker.send_command("runtests", indices=[1])

        assert mock_channel.send.call_count == 3

    def test_send_command_connection_error_includes_command_name(self) -> None:
        mock_channel = MagicMock()
        mock_channel.send.side_effect = EOFError("remote gone")

        worker = RemoteWorker(mock_channel)

        with pytest.raises(ConnectionError) as exc_info:
            worker.send_command("steal", indices=[1, 2, 3])

        assert "steal" in str(exc_info.value)
        assert "remote gone" in str(exc_info.value)