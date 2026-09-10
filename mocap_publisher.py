"""
OptiTrack publisher — runs on the testbed laptop.

Owns the one real connection to Motive (via natnetsdk) and re-broadcasts
every rigid body update over a ZMQ PUB socket bound to the Tailscale
network interface. Every student's subscriber just needs to be connected
to the same Tailscale network — they can do so from anywhere on the
internet, not just from the lab WiFi. They still never touch Motive
directly and don't need natnetsdk installed.

Forwards every rigid body update exactly as received — no filtering.
Any cleaning of corrupted values belongs in each student's own script.

Each message also carries:
  - "name": Motive's own display name for this rigid body (e.g.
    "Rigid Body 001"), fetched once from Motive's data descriptions
    right after connecting — so consumers don't have to guess what a
    streaming ID like 1 or 3 actually refers to.
  - "t_testbed_recv" / "t_publish": this machine's wall-clock time
    when it received the update from Motive and when it was
    published, for latency measurement (see latency_test.py).

Also runs a tiny ZMQ REP server on CLOCK_SYNC_PORT that just echoes
back its own current time — latency_test.py uses this to measure and
correct for clock skew between this machine and the receiver before
trusting any cross-machine latency numbers. (Motive's own per-frame
timestamp is deliberately NOT forwarded: it's Motive's internal clock,
not epoch/wall-clock-aligned with this machine, so comparing it
directly produces meaningless results — confirmed the hard way.)

Run this once, continuously, for the whole class session — students
start and stop their own scripts freely without affecting this.
"""

import json
import socket
import struct
import subprocess
import threading
import time

import zmq
from natnetsdk.NatNetClient import NatNetClient

SERVER_IP = "10.211.223.197"   # Motive's local interface
LOCAL_IP = "10.211.223.199"    # testbed laptop's IP on Motive's network
PUB_PORT = 5556                # reachable to Tailscale-connected students
CLOCK_SYNC_PORT = 5557         # for latency_test.py's clock-offset calibration


def get_tailscale_ip():
    """
    Returns the Tailscale IPv4 address of this machine (the 100.x.x.x address),
    or raises RuntimeError if Tailscale is not running or not connected.

    Tries `tailscale ip -4` first (most reliable). Falls back to scanning
    local interface addresses for a 100.64.0.0/10 address (the CGNAT block
    Tailscale always uses), so it works even if the `tailscale` CLI isn't
    on PATH.
    """
    # Primary: ask the Tailscale CLI directly.
    try:
        out = subprocess.check_output(["tailscale", "ip", "-4"], timeout=3)
        ip = out.decode().strip()
        if ip:
            return ip
    except Exception:
        pass

    # Fallback: scan local interfaces for a 100.64/10 address.
    for family, _, _, _, sockaddr in socket.getaddrinfo(
        socket.gethostname(), None, socket.AF_INET
    ):
        addr = sockaddr[0]
        parts = addr.split(".")
        if len(parts) == 4 and int(parts[0]) == 100 and int(parts[1]) in range(64, 128):
            return addr

    raise RuntimeError(
        "Tailscale IP not found. Is Tailscale installed and connected?\n"
        "Run: tailscale up"
    )

STALE_TIMEOUT = 5.0     # seconds without a rigid body update before reconnecting
RECONNECT_DELAY = 2.0   # seconds to wait between reconnect attempts

NAT_REQUEST_MODELDEF = 4
NAT_MODELDEF = 5


def fetch_rigid_body_names(client, timeout=2.0):
    """
    Asks Motive for its data descriptions (the rigid body ID -> name
    mapping you see in Motive's own UI, e.g. "Rigid Body 001") and
    returns {id: name}. The rigid_body_listener callback only gives an
    ID, not a name, so this is fetched separately, once, after connecting.

    Motive only responds to requests sent from the same socket that
    already completed the NAT_CONNECT handshake — a fresh socket gets
    silently ignored. So this reuses the client's own command_socket to
    send the request (via natnetsdk's own public send_request()), and
    intercepts the response by temporarily wrapping natnetsdk's own
    (private) response parser — the same one its background command
    thread already calls — rather than reading the socket ourselves
    and risking a race with that thread.
    """
    result_holder = {}
    original_unpack = client._NatNetClient__unpack_data_descriptions

    def wrapped_unpack(data, packet_size, major, minor):
        offset, data_descs = original_unpack(data, packet_size, major, minor)
        result_holder["names"] = {
            rb.id_num: rb.sz_name.decode("utf-8", errors="replace")
            for rb in data_descs.rigid_body_list
        }
        return offset, data_descs

    client._NatNetClient__unpack_data_descriptions = wrapped_unpack

    client.send_request(
        client.command_socket,
        client.NAT_REQUEST_MODELDEF,
        "",
        (client.server_ip_address, client.command_port),
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if "names" in result_holder:
            return result_holder["names"]
        time.sleep(0.05)
    return {}


class Publisher:
    def __init__(self):
        tailscale_ip = get_tailscale_ip()
        print(f"[publisher] Tailscale IP: {tailscale_ip}")
        print(f"[publisher] Students should connect to:")
        print(f"            PUB  tcp://{tailscale_ip}:{PUB_PORT}")
        print(f"            REP  tcp://{tailscale_ip}:{CLOCK_SYNC_PORT}")

        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(f"tcp://{tailscale_ip}:{PUB_PORT}")  # Tailscale only — not raw LAN

        self._client = None
        self._last_update = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()

        self._id_to_name = {}  # rigid body streaming ID -> Motive's display name

    def _on_rigid_body(self, new_id, pos, rot):
        t_recv = time.time()  # wall-clock — needed since this gets compared
                               # against the receiver's own wall-clock later
        with self._lock:
            name = self._id_to_name.get(new_id)
        msg = {
            "id": new_id,
            "name": name,                    # e.g. "Rigid Body 001", or None if unknown
            "position": list(pos),
            "quaternion": list(rot),
            "t_testbed_recv": t_recv,        # when the testbed received this from Motive
        }
        msg["t_publish"] = time.time()       # just before sending — isolates any
                                              # processing/serialization delay
                                              # from the network hop itself
        self.pub.send_string(json.dumps(msg))
        with self._lock:
            self._last_update = time.monotonic()

    def _connect(self):
        # natnetsdk's run() spawns its own data/command threads with no
        # daemon flag, so they default to non-daemon. Calling run() from
        # within our own daemon thread makes its child threads inherit
        # daemon=True, so Python never waits on them at process exit —
        # see optitrack_client.py's history for the full explanation.
        result = {}

        def _runner():
            client = NatNetClient()
            client.set_client_address(LOCAL_IP)
            client.set_server_address(SERVER_IP)
            client.set_use_multicast(False)
            client.set_print_level(0)
            client.rigid_body_listener = self._on_rigid_body
            result["started"] = client.run()
            result["client"] = client

        starter = threading.Thread(target=_runner, daemon=True)
        starter.start()
        starter.join()

        if not result.get("started"):
            raise ConnectionError("Failed to open NatNet data/command sockets")

        self._client = result["client"]
        with self._lock:
            self._last_update = time.monotonic()

        try:
            names = fetch_rigid_body_names(self._client)
            with self._lock:
                self._id_to_name = names
        except Exception:
            pass  # messages just carry name=None until a later reconnect succeeds

    def _disconnect(self):
        if self._client is not None:
            client_to_close = self._client
            self._client = None
            threading.Thread(target=self._safe_shutdown, args=(client_to_close,), daemon=True).start()

    @staticmethod
    def _safe_shutdown(client_to_close):
        try:
            client_to_close.shutdown()
        except Exception:
            pass

    def _seconds_since_update(self):
        with self._lock:
            return time.monotonic() - self._last_update

    def run(self):
        while not self._stop.is_set():
            try:
                self._connect()
            except Exception:
                self._stop.wait(RECONNECT_DELAY)
                continue

            while not self._stop.is_set():
                if self._seconds_since_update() > STALE_TIMEOUT:
                    self._disconnect()
                    break
                self._stop.wait(0.5)

        self._disconnect()

    def stop(self):
        self._stop.set()

    def close(self):
        self._disconnect()
        self.pub.close()
        self.ctx.term()


def run_clock_sync_server(stop_event):
    """
    Trivial echo server: replies to any request with this machine's
    current time.time(). latency_test.py uses several round trips to
    this to estimate the clock offset between here and the receiver,
    the same way NTP does, before trusting any cross-machine latency
    numbers computed from timestamps embedded in the PUB stream.
    """
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    # Bind to the same Tailscale IP as the PUB socket so clock-sync is
    # also only reachable over Tailscale, not the raw lab LAN.
    tailscale_ip = get_tailscale_ip()
    sock.bind(f"tcp://{tailscale_ip}:{CLOCK_SYNC_PORT}")
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    try:
        while not stop_event.is_set():
            events = dict(poller.poll(timeout=500))
            if sock in events:
                sock.recv()  # content doesn't matter — any request triggers a reply
                sock.send_string(json.dumps({"server_time": time.time()}))
    finally:
        sock.close()
        ctx.term()


def main():
    publisher = Publisher()
    clock_sync_stop = threading.Event()
    clock_sync_thread = threading.Thread(
        target=run_clock_sync_server, args=(clock_sync_stop,), daemon=True
    )
    clock_sync_thread.start()

    try:
        publisher.run()
    except KeyboardInterrupt:
        publisher.stop()
    finally:
        publisher.close()
        clock_sync_stop.set()


if __name__ == "__main__":
    main()