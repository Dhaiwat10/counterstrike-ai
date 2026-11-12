"""Checks for the weight link between trainers (no engine; run: .venv/bin/python tests/test_peer.py).

Two (then three) PeerSync objects on localhost: published weights arrive, a merge gives the equal-share average,
weights are blended at most once, a peer with the wrong key is refused, and a peer that disappears changes nothing.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.peer import PeerSync  # noqa: E402

PORT, KEY = 47833, b"test-key"


def net(fill):
    m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(fill)
    return m


def wait(cond, what, secs=5.0):
    t = time.time()
    while not cond():
        assert time.time() - t < secs, f"timed out: {what}"
        time.sleep(0.02)


def first(m):
    return next(m.parameters()).flatten()[0].item()


a, b = net(1.0), net(3.0)
A = PeerSync(KEY, listen=PORT, name="a")
B = PeerSync(KEY, connect=[f"127.0.0.1:{PORT}"], name="b")
wait(lambda: len(A.links) == 1 and len(B.links) == 1, "connect")
assert A.merge_into(a, 0)["peer_merged"] == 0, "nothing published yet: nothing to blend"

A.publish(a, 10)
B.publish(b, 11)
wait(lambda: A.links[0].got is not None and B.links[0].got is not None, "exchange")
sa, sb = A.merge_into(a, 12), B.merge_into(b, 11)
assert sa["peer_merged"] == 1 and sa["peer_lag"] == 1 and sb["peer_lag"] == 1, (sa, sb)
assert all(torch.allclose(p, torch.full_like(p, 2.0), atol=1e-3) for m in (a, b) for p in m.parameters()), "average of 1 and 3"
assert abs(sa["peer_diff"] - 2.0) < 1e-3
assert A.merge_into(a, 13)["peer_merged"] == 0 and first(a) == 2.0, "the same weights are never blended twice"

# only the newest published weights count; an older unmerged set is replaced
with torch.no_grad():
    for p in b.parameters():
        p.fill_(10.0)
B.publish(b, 20)
wait(lambda: A.links[0].seen["update"] == 20, "second publish")
A.merge_into(a, 20)
assert abs(first(a) - 6.0) < 1e-3

# a third trainer joins the listener: the hub blends both peers with equal shares
c = net(-3.0)
C = PeerSync(KEY, connect=[f"127.0.0.1:{PORT}"], name="c")
wait(lambda: len([l for l in A.links if l.alive]) == 2, "third trainer")
with torch.no_grad():
    for p in a.parameters():
        p.fill_(0.0)
    for p in b.parameters():
        p.fill_(6.0)
B.publish(b, 30)
C.publish(c, 30)
wait(lambda: all(l.got is not None for l in A.links), "two peers' weights")
assert A.merge_into(a, 30)["peer_merged"] == 2 and abs(first(a) - 1.0) < 1e-3, "(0 + 6 - 3) / 3"

# wrong key: refused, and the listener keeps working
try:
    bad = PeerSync(b"wrong", connect=[f"127.0.0.1:{PORT}"], name="x")
    time.sleep(0.6)
    assert not bad.links, "a peer with the wrong key must not get a link"
    bad.close()
except Exception:
    pass
assert len([l for l in A.links if l.alive]) == 2

# a peer that goes away: its link dies, merges simply find nothing
B.close()
C.close()
wait(lambda: not any(l.alive for l in A.links), "links closing")
before = first(a)
A.publish(a, 40)
s = A.merge_into(a, 40)
assert s["peers"] == 0 and s["peer_merged"] == 0 and first(a) == before
A.close()
print("peer link: exchange, equal-share blend, newest-only, three trainers, wrong key refused, lost peer harmless: all checks passed")
