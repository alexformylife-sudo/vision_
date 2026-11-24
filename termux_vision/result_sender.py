"""输出接口：终端打印与可选非阻塞 TCP，发送策略与通道解耦。"""
from __future__ import annotations

import logging
import runpy
import socket
import sys
import threading
import time
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("_bootstrap.py")), run_name="termux_vision_bootstrap")

from termux_vision.config import Config
from termux_vision.util import interruptible_sleep

LOGGER = logging.getLogger(__name__)


class ConsoleSender:
    def send(self, message: str) -> None:
        sys.stdout.write(message)
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.flush()


class TcpSender:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._latest_message: str | None = None
        self._link_down = False  # 最近一次连接或发送失败时为 True，关闭时不再白等
        self._running = True
        self._socket: socket.socket | None = None
        self._thread = threading.Thread(target=self._send_loop, name="tcp-sender", daemon=True)
        self._thread.start()
        LOGGER.info("TCP 发送已启动 %s:%s", config.tcp_host, config.tcp_port)

    def send(self, message: str) -> None:
        with self._lock:
            self._latest_message = message

    def close(self) -> None:
        self._wait_for_flush()
        with self._lock:
            self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=self._config.thread_join_timeout_s)
        self._close_socket()

    def _wait_for_flush(self) -> None:
        deadline = time.monotonic() + self._config.thread_join_timeout_s
        while time.monotonic() < deadline:
            if not self._has_deliverable():
                return
            time.sleep(self._config.idle_poll_s)
        LOGGER.warning("TCP 关闭前仍有未发出报文，停止等待")

    def _has_deliverable(self) -> bool:
        with self._lock:
            return self._latest_message is not None and not self._link_down

    def _mark_link_down(self) -> None:
        with self._lock:
            self._link_down = True

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    def _send_loop(self) -> None:
        while self._is_running():
            message = self._take_message()
            if message is None:
                interruptible_sleep(self._is_running, self._config.idle_poll_s, self._config.idle_poll_s)
                continue
            self._send_or_restore(message)

    def _take_message(self) -> str | None:
        with self._lock:
            message = self._latest_message
            self._latest_message = None
            return message

    def _restore_if_empty(self, message: str) -> None:
        with self._lock:
            if self._latest_message is None:
                self._latest_message = message

    def _send_or_restore(self, message: str) -> None:
        client = self._connected_socket()
        if client is None:
            self._restore_if_empty(message)
            return
        try:
            client.sendall(message.encode("utf-8"))
        except (TimeoutError, OSError, BrokenPipeError) as exc:
            LOGGER.warning("TCP 发送失败: %s", exc)
            self._mark_link_down()
            self._close_socket()
            self._restore_if_empty(message)

    def _connected_socket(self) -> socket.socket | None:
        with self._lock:
            if self._socket is not None:
                return self._socket
        if not self._connect():
            return None
        with self._lock:
            return self._socket

    def _connect(self) -> bool:
        try:
            new_socket = socket.create_connection(
                (self._config.tcp_host, self._config.tcp_port),
                timeout=self._config.tcp_connect_timeout_s,
            )
            new_socket.settimeout(self._config.tcp_send_timeout_s)
        except (TimeoutError, OSError) as exc:
            LOGGER.warning("TCP 连接失败: %s", exc)
            self._mark_link_down()
            interruptible_sleep(self._is_running, self._config.reconnect_delay_s, self._config.idle_poll_s)
            return False
        return self._store_socket(new_socket)

    def _store_socket(self, new_socket: socket.socket) -> bool:
        with self._lock:
            if not self._running:
                new_socket.close()
                return False
            self._socket = new_socket
            self._link_down = False
            return True

    def _close_socket(self) -> None:
        with self._lock:
            client = self._socket
            self._socket = None
        if client is None:
            return
        _shutdown_and_close(client)


class CompositeSender:
    def __init__(self, senders: list) -> None:
        self._senders = senders

    def send(self, message: str) -> None:
        for sender in self._senders:
            sender.send(message)

    def close(self) -> None:
        for sender in self._senders:
            sender.close()


class ResultSender:
    def __init__(self, inner, send_policy: str, heartbeat_interval_s: float) -> None:
        self._inner = inner
        self._send_policy = send_policy
        self._heartbeat_interval_s = heartbeat_interval_s
        self._last_message: str | None = None
        self._last_sent_at = 0.0

    def send(self, message: str) -> None:
        now = time.monotonic()
        if not self._should_send(message, now):
            return
        self._inner.send(message)
        self._last_message = message
        self._last_sent_at = now

    def close(self) -> None:
        self._inner.close()

    def _should_send(self, message: str, now: float) -> bool:
        if self._last_message is None or message != self._last_message:
            return True
        if self._send_policy != "heartbeat":
            return False
        elapsed = now - self._last_sent_at
        return elapsed >= self._heartbeat_interval_s


def build_sender(config: Config) -> ResultSender:
    senders = []
    if config.print_enabled:
        senders.append(ConsoleSender())
    if config.tcp_enabled:
        senders.append(TcpSender(config))
    if not senders:
        raise ValueError("未启用任何输出：请打开终端打印或 TCP")
    inner = senders[0] if len(senders) == 1 else CompositeSender(senders)
    return ResultSender(inner, config.send_policy, config.heartbeat_interval_s)


def _shutdown_and_close(client: socket.socket) -> None:
    try:
        client.shutdown(socket.SHUT_RDWR)
    except OSError as exc:
        LOGGER.debug("TCP shutdown 跳过: %s", exc)
    try:
        client.close()
    except OSError as exc:
        LOGGER.warning("TCP 关闭套接字失败: %s", exc)
