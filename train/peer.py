"""Link between trainers on different machines: each keeps its own games and its own learner, and after an update
blends in the other's latest weights (gossip averaging). Only weights cross the network (~28 MB per exchange for the
14M-parameter policy), so home Wi-Fi is enough; frames never leave the machine they were rendered on.

One machine listens (--peer-listen PORT), the others connect to it (--peer-connect HOST:PORT). Every connection
carries weights both ways. The trainer never waits for the network: two background threads per connection send the
newest published weights and keep the newest received ones; `merge_into` uses whatever has arrived. A peer that is
slow, behind or gone only means nothing new to blend.

Wire format (multiprocessing.connection, HMAC-authenticated with a shared key file): a JSON header
{"update": n, "name": str} as bytes, then the weights as a torch-saved dict of half-precision tensors, loaded with
weights_only=True. Nothing received is ever unpickled as an object.
"""
import io
import json
import threading
import time
from multiprocessing.connection import Client, Listener

import torch


def pack(model, update, name):
    """(header bytes, blob bytes) for the model's floating-point weights, in half precision."""
    sd = {k: v.detach().to("cpu", torch.float16) for k, v in model.state_dict().items() if v.is_floating_point()}
    buf = io.BytesIO()
    torch.save(sd, buf)
    return json.dumps(dict(update=int(update), name=name)).encode(), buf.getvalue()


def unpack(blob):
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


class _Link:
    """One connection: a sender thread (newest published weights, at most once each) and a receiver thread."""

    def __init__(self, conn, owner):
        self.conn, self.owner = conn, owner
        self.got = None        # (header dict, blob) newest received, not yet merged
        self.seen = None       # header of the newest received (merged or not)
        self.sent = -1         # update number of the last weights sent
        self.alive = True
        self.lock = threading.Lock()
        for fn in (self._send, self._recv):
            threading.Thread(target=fn, daemon=True).start()

    def _send(self):
        try:
            while self.alive and not self.owner.closed:
                pub = self.owner.published
                if pub is None or pub[0] == self.sent:
                    time.sleep(0.05)
                    continue
                self.conn.send_bytes(pub[1])
                self.conn.send_bytes(pub[2])
                self.sent = pub[0]
        except (OSError, EOFError, ValueError):
            pass
        self.alive = False

    def _recv(self):
        try:
            while self.alive and not self.owner.closed:
                head = json.loads(self.conn.recv_bytes(1 << 16))
                blob = self.conn.recv_bytes(1 << 30)
                with self.lock:
                    self.got, self.seen = (head, blob), head
        except (OSError, EOFError, ValueError):
            pass
        self.alive = False

    def take(self):
        with self.lock:
            got, self.got = self.got, None
        return got


class PeerSync:
    def __init__(self, key, listen=0, connect=(), name="", every=1):
        """key: shared secret (bytes). listen: port to accept peers on (0 = none). connect: ["host:port", ...]."""
        assert key and (listen or connect)
        self.key, self.name, self.every = key, name, max(1, int(every))
        self.published = None  # (update, header bytes, blob bytes)
        self.links, self.closed = [], False
        self.lock = threading.Lock()
        self.listener = None
        if listen:
            self.listener = Listener(("0.0.0.0", int(listen)), authkey=key)
            threading.Thread(target=self._accept, daemon=True).start()
        for target in connect:
            threading.Thread(target=self._dial, args=(target,), daemon=True).start()

    def _add(self, conn):
        with self.lock:
            self.links = [l for l in self.links if l.alive] + [_Link(conn, self)]

    def _accept(self):
        while not self.closed:
            try:
                self._add(self.listener.accept())
            except Exception:  # wrong key, a dropped handshake, or the listener closing: keep serving the others
                if self.closed:
                    return
                time.sleep(0.5)

    def _dial(self, target):
        host, port = target.rsplit(":", 1)
        link = None
        while not self.closed:  # (re)connect for as long as the trainer runs
            if link is None or not link.alive:
                try:
                    conn = Client((host, int(port)), authkey=self.key)
                    self._add(conn)
                    link = self.links[-1]
                except Exception:
                    link = None
            time.sleep(2.0)

    def publish(self, model, update):
        head, blob = pack(model, update, self.name)
        self.published = (int(update), head, blob)

    @torch.no_grad()
    def merge_into(self, model, update):
        """Blend every peer's newest unmerged weights into `model` (equal shares). Returns stats for the log."""
        with self.lock:
            links = [l for l in self.links if l.alive]
        new = [g for g in (l.take() for l in links) if g is not None]
        stats = dict(peers=len(links), peer_merged=len(new))
        if not new:
            return stats
        own = model.state_dict()
        theirs = [unpack(blob) for _, blob in new]
        diff, count = 0.0, 0
        for k, v in own.items():
            ts = [t[k] for t in theirs if k in t and t[k].shape == v.shape]
            if not v.is_floating_point() or len(ts) != len(theirs):
                continue  # a tensor some peer does not have (different heads): leave ours alone
            pull = torch.zeros_like(v)
            for t in ts:
                d = t.to(v.device, v.dtype) - v
                diff += d.abs().mean().item()
                count += 1
                pull += d
            v.add_(pull, alpha=1.0 / (len(ts) + 1))  # equal shares: (ours + sum of theirs) / (n + 1)
        stats["peer_lag"] = max(int(update) - int(head["update"]) for head, _ in new)
        stats["peer_diff"] = diff / max(count, 1)
        return stats

    def step(self, model, update):
        """Call once after every update: publish (every `every` updates), then blend what has arrived."""
        if update % self.every == 0:
            self.publish(model, update)
        return self.merge_into(model, update)

    def close(self):
        self.closed = True
        if self.listener is not None:
            try:
                self.listener.close()
            except OSError:
                pass
        for l in self.links:
            l.alive = False
            try:
                l.conn.close()
            except OSError:
                pass
