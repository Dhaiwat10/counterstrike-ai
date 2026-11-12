"""Central learner link: one machine learns full-time (the LEARNER), others run the games (ACTORS).

    learner:  python -m train.ppo --collect remote --actor-listen 47900 --envs 12 ...   (no games on this machine)
    actor:    python -m train.actor --connect LEARNER_HOST:47900 --envs 12            (games + a copy of the model)

The actor plays with the newest weights it has received, ships every batch of finished segments (the same tensors the
local Collector returns), and swaps in refreshed weights between batches. The learner's RemoteCollector has the
Collector's contract (`collect(n, on_episode)`, `.beta`, `set_knobs`), so the PPO loop does not change.

Wire format: multiprocessing.connection over TCP, HMAC-authenticated with the shared key file (data/peer.key);
send_bytes/recv_bytes only. Every message is a JSON header, optionally followed by one binary blob. Rollouts use the
self-describing raw format below (numpy buffers, whitelisted dtypes); weights use torch.save of half-precision tensors,
loaded with weights_only=True. Nothing received is ever unpickled as an object.

Rollout blob: uint32 header length, JSON header (arrays: key, dtype, shape, codec, chunk sizes; episodes; scalars),
then the chunks. uint8 image tensors [n, T, ...] are coded per segment as a frame-to-frame difference along T (mod 256,
exactly reversible) and zlib level 1; everything else is stored raw. Chunks are (de)compressed in a thread pool (numpy
and zlib release the GIL), and on the actor in its sender thread, so the games are not held up by it.

Back-pressure: the actor drops a new batch while the previous one is still being encoded/sent; the learner keeps at
most 2 waiting batches (dropping the oldest). Episode records of a dropped batch are never lost: they travel with the
next batch (actor) or are handed out with the next collect (learner), so curriculum statistics see every round once.
"""
from __future__ import annotations

import io
import json
import math
import os
import socket
import struct
import threading
import time
import uuid
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Connection, answer_challenge, deliver_challenge

import numpy as np
import torch

MAX_HEADER = 1 << 24
MAX_BLOB = (1 << 31) - 1
MAX_WAITING = 2           # learner: batches allowed to wait; one more drops the oldest
PING_EVERY = 5.0          # learner -> actor keep-alive when there is nothing else to send (s)
RX_TIMEOUT = 60.0         # actor: no message from the learner for this long -> the link is dead, reconnect
DTYPES = {"uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "bool"}

_pool = None
_pool_lock = threading.Lock()


def _workers():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max(1, min(4, (os.cpu_count() or 2) // 2)), thread_name_prefix="remote-codec")
        return _pool


# ---------------------------------------------------------------------------------- rollout coding
def _jsonable(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    raise TypeError(f"cannot send {type(x).__name__} in an episode record")


def _delta_z(row):
    """[T, ...] uint8 -> zlib(first frame, then frame-to-frame differences mod 256)."""
    d = np.empty_like(row)
    if len(row):
        d[0] = row[0]
        np.subtract(row[1:], row[:-1], out=d[1:])  # uint8 arithmetic wraps mod 256
    return zlib.compress(d, 1)


def _undelta(data, out):
    raw = zlib.decompress(data)
    if len(raw) != out.nbytes:
        raise ValueError("corrupt rollout chunk")
    np.cumsum(np.frombuffer(raw, np.uint8).reshape(out.shape), axis=0, dtype=np.uint8, out=out)


def encode_rollout(ro, episodes=(), meta=None) -> bytes:
    """Collector output (tensors on any device, the h0 tuple, scalars) + episode records -> bytes. Lossless."""
    specs, jobs, scalars, order = [], [], {}, []
    for k, v in ro.items():
        order.append(k)
        if torch.is_tensor(v):
            items = [(k, v)]
        elif isinstance(v, tuple) and v and all(torch.is_tensor(x) for x in v):
            items = [(f"{k}#{i}", x) for i, x in enumerate(v)]
            scalars[k] = {"tuple": len(v)}
        elif v is None or isinstance(v, (bool, int, float, str)):
            scalars[k] = {"value": v}
            continue
        else:
            raise TypeError(f"rollout key {k}: cannot send {type(v).__name__}")
        for name, x in items:
            a = x.detach().cpu().contiguous().numpy()
            if a.dtype.name not in DTYPES:
                raise TypeError(f"rollout key {name}: dtype {a.dtype} not supported")
            if a.dtype == np.uint8 and a.ndim >= 2:
                spec = dict(k=name, dtype="uint8", shape=list(a.shape), codec="dz")
                spec["_jobs"] = [len(jobs) + i for i in range(a.shape[0])]
                jobs += [(_delta_z, a[i]) for i in range(a.shape[0])]
            else:
                spec = dict(k=name, dtype=a.dtype.name, shape=list(a.shape), codec="raw")
                spec["_jobs"] = [len(jobs)]
                jobs.append((None, a))
            specs.append(spec)
    big = [i for i, (fn, _) in enumerate(jobs) if fn is not None]
    chunks = [a.tobytes() if fn is None else None for fn, a in jobs]
    for i, c in zip(big, _workers().map(lambda j: jobs[j][0](jobs[j][1]), big)):
        chunks[i] = c
    for s in specs:
        s["chunks"] = [len(chunks[j]) for j in s.pop("_jobs")]
    crc = 0
    for c in chunks:
        crc = zlib.crc32(c, crc)
    head = json.dumps(dict(arrays=specs, scalars=scalars, order=order, episodes=list(episodes), meta=meta or {},
                           crc=crc), default=_jsonable).encode()
    return b"".join([struct.pack("<I", len(head)), head] + chunks)


def _head(blob):
    mv = memoryview(blob)
    n = struct.unpack_from("<I", mv, 0)[0]
    if n > MAX_HEADER or 4 + n > len(mv):
        raise ValueError("corrupt rollout header")
    return json.loads(bytes(mv[4:4 + n])), mv, 4 + n


def rollout_episodes(blob):
    """Only the episode records (and meta) of an encoded rollout, without decoding its tensors."""
    head, _, _ = _head(blob)
    return head["episodes"], head.get("meta", {})


def decode_rollout(blob, device="cpu", meta_out=None):
    """bytes -> (rollout dict with tensors on `device`, episode records). Exact inverse of encode_rollout."""
    head, mv, off = _head(blob)
    if zlib.crc32(mv[off:]) != head.get("crc"):
        raise ValueError("rollout checksum mismatch")
    arrays, jobs = {}, []
    for s in head["arrays"]:
        if s["dtype"] not in DTYPES or s["codec"] not in ("raw", "dz"):
            raise ValueError(f"unsupported array {s['k']}: {s['dtype']}/{s['codec']}")
        shape = tuple(int(x) for x in s["shape"])
        out = np.empty(shape, np.dtype(s["dtype"]))
        if s["codec"] == "raw":
            (n,) = s["chunks"]
            if n != out.nbytes or off + n > len(mv):
                raise ValueError(f"corrupt array {s['k']}")
            out.reshape(-1).view(np.uint8)[:] = np.frombuffer(mv[off:off + n], np.uint8)
            off += n
        else:
            if len(s["chunks"]) != shape[0]:
                raise ValueError(f"corrupt array {s['k']}")
            for i, n in enumerate(s["chunks"]):
                if off + n > len(mv):
                    raise ValueError(f"corrupt array {s['k']}")
                jobs.append((mv[off:off + n], out[i]))
                off += n
        arrays[s["k"]] = out
    list(_workers().map(lambda j: _undelta(*j), jobs))
    if off != len(mv):
        raise ValueError("trailing bytes in rollout")
    ro = {}
    for k in head["order"]:
        sc = head["scalars"].get(k)
        if sc is None:
            ro[k] = torch.from_numpy(arrays[k]).to(device)
        elif "tuple" in sc:
            ro[k] = tuple(torch.from_numpy(arrays[f"{k}#{i}"]).to(device) for i in range(sc["tuple"]))
        else:
            ro[k] = sc["value"]
    if meta_out is not None:
        meta_out.update(head.get("meta", {}))
    return ro, head["episodes"]


# ---------------------------------------------------------------------------------- weights
def pack_weights(sd_half):
    buf = io.BytesIO()
    torch.save(sd_half, buf)
    return buf.getvalue()


def half_state(model):
    """Snapshot of the model's floating-point weights in half precision, on the CPU."""
    return {k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items() if v.is_floating_point()}


def unpack_weights(blob):
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------------- transport
def _sock(conn):
    return socket.socket(fileno=os.dup(conn.fileno()))


def _tune(conn):
    """TCP keep-alive (a silently vanished peer is noticed in ~1 min) and big buffers for the 30-100 MB messages."""
    try:
        s = _sock(conn)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            for opt, val in (("TCP_KEEPALIVE", 20), ("TCP_KEEPIDLE", 20), ("TCP_KEEPINTVL", 5), ("TCP_KEEPCNT", 6)):
                if hasattr(socket, opt):
                    s.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
                s.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)
        finally:
            s.close()
    except OSError:
        pass


def _timeouts(conn, secs):
    """Kernel send/receive timeouts on a blocking socket (0 = none): a read or write that makes no progress for this
    long fails with an OSError instead of hanging (Python-level socket timeouts would make the fd non-blocking, which
    multiprocessing's Connection cannot use)."""
    tv = struct.pack("ll", int(secs), int((secs - int(secs)) * 1e6))
    s = _sock(conn)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, tv)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, tv)
    finally:
        s.close()


def _shutdown(conn):
    """Wake any thread blocked on this connection, then close it."""
    try:
        s = _sock(conn)
        try:
            s.shutdown(socket.SHUT_RDWR)
        finally:
            s.close()
    except OSError:
        pass
    try:
        conn.close()
    except OSError:
        pass


def _send(conn, head, blob=None):
    conn.send_bytes(json.dumps(dict(head, blob=blob is not None)).encode())
    if blob is not None:
        conn.send_bytes(blob)


def _recv(conn):
    head = json.loads(conn.recv_bytes(MAX_HEADER))
    if not isinstance(head, dict):
        raise ValueError("bad message")
    return head, (conn.recv_bytes(MAX_BLOB) if head.get("blob") else None)


def dial(target, key, timeout=5.0):
    """Connect to HOST:PORT and authenticate (raises AuthenticationError on a wrong key)."""
    host, port = target.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=timeout)
    s.setblocking(True)
    conn = Connection(s.detach())
    try:
        _timeouts(conn, timeout)  # the handshake must not hang on a half-open connection
        answer_challenge(conn, key)
        deliver_challenge(conn, key)
    except BaseException:
        _shutdown(conn)
        raise
    _tune(conn)
    return conn


# ---------------------------------------------------------------------------------- learner side
class _Link:
    """Learner end of one actor connection: handshake, a sender thread (control, newest weights, pings), receiver."""

    def __init__(self, conn, owner):
        self.conn, self.owner = conn, owner
        self.alive, self.name, self.id, self.error = True, "?", None, None
        self.sent_w, self.sent_ctl = None, None
        threading.Thread(target=self._run, daemon=True).start()

    def close(self):
        self.alive = False
        _shutdown(self.conn)

    def _run(self):
        try:
            _timeouts(self.conn, 30)  # a client that connects and then stalls only holds up its own link
            deliver_challenge(self.conn, self.owner.key)
            answer_challenge(self.conn, self.owner.key)
            _tune(self.conn)
            head, _ = _recv(self.conn)
            if head.get("type") != "hello":
                raise ValueError("expected hello")
            self.name, self.id = str(head.get("name", "?")), str(head.get("id"))
            _timeouts(self.conn, 120)  # the config carries the weights (~28 MB)
            cfg, ctl_v, (w_upd, w_blob) = self.owner._snapshot()
            _send(self.conn, dict(type="config", config=cfg, update=w_upd), w_blob)
            # data phase: a send that makes no progress for 60 s means the actor is gone; reads may wait any time
            # (an actor rebuilding its games is silent for a while)
            _timeouts(self.conn, 0)
            s = _sock(self.conn)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, struct.pack("ll", 60, 0))
            s.close()
            self.sent_w, self.sent_ctl = w_upd, ctl_v
            self.owner._register(self)
            threading.Thread(target=self._send_loop, daemon=True).start()
            while self.alive and not self.owner.closed:
                head, blob = _recv(self.conn)
                if head.get("type") == "batch" and blob is not None:
                    self.owner._incoming(self, head, blob)
        except Exception as e:  # wrong key, the actor going away, a broken message: this link ends, the others go on
            self.error = self.error or repr(e)
        self._end()

    def _send_loop(self):
        last = time.time()
        try:
            while self.alive and not self.owner.closed:
                did = False
                ctl_v, ctl = self.owner._ctl
                if ctl_v != self.sent_ctl:
                    _send(self.conn, dict(type="control", **ctl))
                    self.sent_ctl, did = ctl_v, True
                w = self.owner._weights()
                if w is not None and w[0] != self.sent_w:
                    _send(self.conn, dict(type="weights", update=w[0]), w[1])
                    self.sent_w, did = w[0], True
                if did:
                    last = time.time()
                elif time.time() - last > PING_EVERY:
                    _send(self.conn, dict(type="ping"))
                    last = time.time()
                else:
                    time.sleep(0.02)
        except Exception as e:
            self.error = self.error or repr(e)
        self._end()

    def _end(self):
        was = self.alive
        self.close()
        if self.id is not None and was and not self.owner.closed:
            self.owner.log(f"[remote] actor {self.name} disconnected ({self.error})", flush=True)


class RemoteCollector:
    """Learner-side stand-in for Collector: batches come from actors over the network.

    config: dict the actors need to build identical games and model (see train/ppo.py, --collect remote).
    """

    def __init__(self, port, key, config, model, update, dev, host="0.0.0.0", log=print):
        assert key
        self.key, self.dev, self.log = key, torch.device(dev), log
        self.config = dict(config)
        self.closed = False
        self.cond = threading.Condition()
        self.links = []
        self._raw, self._ready = deque(), deque()   # waiting batches: received (encoded) / decoded
        self._eps = []                                # episode records of dropped batches, handed out with the next one
        self._ctl = (0, dict(knobs=self.config.get("knobs"), beta=float(self.config.get("beta", 0.0))))
        self._beta = self._ctl[1]["beta"]
        self._pub, self._blob = None, None            # (update, half state dict), (update, bytes)
        self._wlock = threading.Lock()
        self.update = int(update)
        self.n_recv = self.n_used = self.n_drop = 0
        self.mb = 0.0
        self.wait_total = self.wait_last = self.decode_last = 0.0
        self.age_last = None
        self.actor_drops, self.actor_stats = {}, {}
        self.last_head = None
        self.publish(model, update)
        self.server = socket.create_server((host, int(port)))  # SO_REUSEADDR: a restarted learner gets its port back
        self.server.settimeout(0.5)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()
        threading.Thread(target=self._decoder, daemon=True).start()

    # -- trainer-facing API
    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        value = float(value)
        if value != self._beta:
            self._beta = value
            v, ctl = self._ctl
            self._ctl = (v + 1, dict(ctl, beta=value))

    def set_knobs(self, knobs):
        v, ctl = self._ctl
        self.config["knobs"] = dict(knobs)
        self._ctl = (v + 1, dict(ctl, knobs=dict(knobs)))

    def publish(self, model, update):
        """Newest weights for every actor. Only a CPU half-precision copy is made here; serialisation and sending
        happen in the link threads, and an actor that is still receiving older weights simply skips to the newest."""
        sd = half_state(model)
        with self._wlock:
            self._pub = (int(update), sd)
        self.update = int(update)

    def collect(self, n, on_episode, timeout=None):
        """The oldest waiting batch (blocks until one arrives; `timeout` s -> TimeoutError). Episode records of batches
        dropped since the last call are passed to on_episode first, then this batch's."""
        t = time.time()
        warned = t
        with self.cond:
            while not self._ready:
                if self.closed:
                    raise RuntimeError("remote collector closed")
                if timeout is not None and time.time() - t > timeout:
                    raise TimeoutError("no batch from the actors")
                self.cond.wait(1.0)
                if time.time() - warned > 60:
                    warned = time.time()
                    self.log(f"[remote] waiting for actor batches ({len([l for l in self.links if l.alive])} actors "
                             f"connected, {time.time() - t:.0f} s)", flush=True)
            head, ro, eps = self._ready.popleft()
            older, self._eps = self._eps, []
        self.wait_last = time.time() - t
        self.wait_total += self.wait_last
        assert ro["P"].shape[0] == n, f"actor sent {ro['P'].shape[0]} segments, expected {n}"
        out = {k: (tuple(x.to(self.dev) for x in v) if isinstance(v, tuple) else
                   v.to(self.dev) if torch.is_tensor(v) else v) for k, v in ro.items()}
        self.n_used += 1
        self.last_head = head
        self.age_last = self.update - int(head.get("wupdate", self.update))
        for e in older + eps:
            on_episode(e)
        return out

    def stats(self):
        with self.cond:
            q = len(self._raw) + len(self._ready)
        st = list(self.actor_stats.values())
        return dict(remote_batches=self.n_recv, remote_used=self.n_used, remote_dropped=self.n_drop,
                    actor_dropped=int(sum(self.actor_drops.values())), remote_queue=q,
                    remote_wait_s=round(self.wait_last, 3), remote_wait_total_s=round(self.wait_total, 1),
                    remote_mb=round(self.mb, 1), weights_age=self.age_last,
                    actors=len([l for l in self.links if l.alive]), remote_decode_s=round(self.decode_last, 3),
                    actor_encode_s=round(max(s.get("enc_s", 0.0) for s in st), 3) if st else None,
                    actor_send_s=round(max(s.get("send_s", 0.0) for s in st), 3) if st else None)

    def close(self):
        self.closed = True
        try:
            self.server.close()
        except OSError:
            pass
        for link in list(self.links):
            link.close()
        with self.cond:
            self.cond.notify_all()

    # -- internals
    def _snapshot(self):
        return dict(self.config, knobs=self._ctl[1]["knobs"], beta=self._ctl[1]["beta"]), self._ctl[0], self._weights()

    def _weights(self):
        with self._wlock:
            if self._pub is None:
                return None
            if self._blob is None or self._blob[0] != self._pub[0]:
                self._blob = (self._pub[0], pack_weights(self._pub[1]))
            return self._blob

    def _register(self, link):
        with self.cond:
            for old in self.links:
                if old is not link and old.id == link.id:
                    old.close()  # the same actor reconnected: its previous connection is dead
            self.links = [l for l in self.links if l.alive and l is not link] + [link]
        self.log(f"[remote] actor {link.name} connected ({len(self.links)} total)", flush=True)

    def _accept(self):
        while not self.closed:
            try:
                s, _ = self.server.accept()
            except (socket.timeout, TimeoutError):
                continue
            except OSError:
                if self.closed:
                    return
                time.sleep(0.2)
                continue
            s.settimeout(None)
            _Link(Connection(s.detach()), self)

    def _incoming(self, link, head, blob):
        with self.cond:
            self.n_recv += 1
            self.mb += len(blob) / 1e6
            self.actor_drops[link.id] = int(head.get("drops", 0))
            self.actor_stats[link.id] = dict(enc_s=float(head.get("enc_s", 0.0)), send_s=float(head.get("send_s", 0.0)))
            self._raw.append((head, blob))
            self._trim()
            self.cond.notify_all()

    def _trim(self):  # (holding self.cond) keep at most MAX_WAITING batches; the oldest go, their rounds are kept
        while len(self._raw) + len(self._ready) > MAX_WAITING:
            if self._ready:
                _, _, eps = self._ready.popleft()
            else:
                eps, _ = rollout_episodes(self._raw.popleft()[1])
            self._eps.extend(eps)
            self.n_drop += 1

    def _decoder(self):
        while not self.closed:
            with self.cond:
                while not self._raw and not self.closed:
                    self.cond.wait(0.5)
                if self.closed:
                    return
                head, blob = self._raw.popleft()
            t = time.time()
            try:
                ro, eps = decode_rollout(blob, "cpu")
            except Exception as e:  # a broken batch is skipped, never trained on
                self.log(f"[remote] dropped an undecodable batch: {e}", flush=True)
                continue
            self.decode_last = time.time() - t
            with self.cond:
                self._ready.append((head, ro, eps))
                self._trim()
                self.cond.notify_all()


# ---------------------------------------------------------------------------------- actor side
class ActorLink:
    """Actor end of the connection: handshake (config + initial weights), a receiver thread that keeps the newest
    weights and control message, and a sender thread that encodes and ships one batch at a time."""

    def __init__(self, target, key, name="actor", actor_id=None, timeout=5.0):
        self.conn = dial(target, key, timeout)
        self.alive, self.error = True, None
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.slot = None               # batch waiting for (or in) the sender
        self.busy = False
        self.n_sent, self.mb_sent, self.enc_s, self.send_s = 0, 0.0, 0.0, 0.0
        self.send_delay = 0.0          # (tests only) extra seconds per send
        try:
            _send(self.conn, dict(type="hello", name=name, id=actor_id or uuid.uuid4().hex))
            _timeouts(self.conn, 120)  # the config carries the weights (~28 MB)
            head, blob = _recv(self.conn)
            if head.get("type") != "config" or blob is None:
                raise ValueError("expected config")
            # data phase: the learner sends at least a ping every PING_EVERY s; no progress for 60 s = link dead
            _timeouts(self.conn, RX_TIMEOUT)
        except BaseException:
            _shutdown(self.conn)
            raise
        self.config = head["config"]
        self._weights = (int(head["update"]), blob)
        self._control = None
        self.last_rx = time.time()
        for fn in (self._recv_loop, self._send_loop):
            threading.Thread(target=fn, daemon=True).start()

    def ok(self, timeout=RX_TIMEOUT):
        return self.alive and time.time() - self.last_rx < timeout

    def take_weights(self):
        """(update, half state dict) of the newest weights received since the last call, or None."""
        with self.lock:
            w, self._weights = self._weights, None
        return None if w is None else (w[0], unpack_weights(w[1]))

    def take_control(self):
        with self.lock:
            c, self._control = self._control, None
        return c

    def offer(self, ro, episodes, head, wait=False):
        """Hand a batch to the sender. While the previous batch is still being encoded/sent: with wait=False return
        False (the caller drops the batch); with wait=True block until the sender is free, so the games are paced by
        the link instead of producing batches that get thrown away (collecting a batch and sending one take about the
        same time over Wi-Fi, where "drop if busy" would discard every second batch). False when the link is down."""
        while True:
            with self.lock:
                if not self.alive:
                    return False
                if not self.busy:
                    self.busy, self.slot = True, (ro, list(episodes), dict(head))
                    break
                if not wait:
                    return False
            time.sleep(0.01)
        self.wake.set()
        return True

    def idle(self):
        return not self.busy

    def close(self):
        self.alive = False
        self.wake.set()
        _shutdown(self.conn)

    def _recv_loop(self):
        try:
            while self.alive:
                head, blob = _recv(self.conn)
                self.last_rx = time.time()
                kind = head.get("type")
                with self.lock:
                    if kind == "weights" and blob is not None:
                        self._weights = (int(head["update"]), blob)
                    elif kind == "control":
                        c = dict(self._control or {})
                        c.update({k: head[k] for k in ("knobs", "beta") if k in head})
                        self._control = c
        except Exception as e:
            self.error = self.error or repr(e)
        self.close()

    def _send_loop(self):
        try:
            while self.alive:
                self.wake.wait(0.5)
                self.wake.clear()
                with self.lock:
                    item, self.slot = self.slot, None
                if item is None:
                    continue
                ro, eps, head = item
                t = time.time()
                blob = encode_rollout(ro, eps)
                enc = time.time() - t
                del ro, item
                if self.send_delay:
                    time.sleep(self.send_delay)
                t = time.time()
                _send(self.conn, dict(head, type="batch", enc_s=enc, send_s=self.send_s), blob)
                self.enc_s, self.send_s = enc, time.time() - t
                self.n_sent += 1
                self.mb_sent += len(blob) / 1e6
                with self.lock:
                    self.busy = False
        except Exception as e:
            self.error = self.error or repr(e)
        self.close()
