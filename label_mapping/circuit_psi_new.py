"""
Fuzzy circuit-PSI relation table: server-free, semi-honest (attn_filter / precision).

Same function as rt_protocol.run_rt_protocol(method="attn_filter", attn_match="precision"),
but NO party learns any rank, match bit, edge or group size while the table is built.
The aggregation server is not involved until the end; it only receives the final
partition of masked ids (|global_id|client_id|masked_id|).

Parties
  every client  Phase 1 locally (keyword attention z, affinity profile s, features v).
  client pair   Phase 2 fuzzy circuit-PSI: BFV inner products -> additive shares mod t.
                Receiver j owns its BFV key and sends Enc([z|s|v]) densely packed (one message
                for all signals); sender i returns Enc(x*y + rho) per slot, keeps -rho.
  P0, P1        two compute clients holding 2-of-2 additive shares over Z_2^64.
  helper        a third client; sends correlated randomness only (Beaver triples, random
                bits); never sees shares or opened values.  Semi-honest, non-colluding.
  server        receives P0/P1 shares of the final component labels, reconstructs the table.

Phase 3 circuit (all on shares): ladder ranks -> rank bounds -> precision rule (floor, rival
margin) -> Mutual -> oblivious sort of candidate edges by weight -> union-find with
same-client and complete-link support constraints.  Output identical to the plaintext
global_table (same masked ids, same gids) up to fixed-point quantization of cosines.

Network is simulated in-process; shares, masks and HE ciphertexts are real. Byte counts
are reported in diag.
"""
import secrets
import time
from collections import defaultdict

import numpy as np

from label_mapping import rt_protocol as rp

POLY = 8192                 # BFV ring degree = slots per ciphertext
QSCALE = 2 ** 14            # q = round(x * QSCALE): |<q,q'>| <~ 2^28 < t/2; cos error ~3e-5
FIX = 10 ** 6               # fixed-point for ladder values (ladders are rounded to 6 decimals)
U64 = np.uint64
ONE = U64(1)


# =============================================================== small helpers
def _is_prime(n):
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in bases:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _bfv_prime(bits=30):
    """Largest prime < 2^bits with t = 1 mod 2*POLY (BFV batching)."""
    step = 2 * POLY
    t = (1 << bits) // step * step + 1
    while not _is_prime(t):
        t -= step
    return t


T = _bfv_prime()            # 1073692673


def enc(x):
    """public signed ints -> uint64 two's complement"""
    return np.asarray(x, dtype=np.int64).astype(U64)


def dec(x):
    return np.asarray(x, dtype=U64).astype(np.int64)


# =============================================================== 2PC + helper over Z_2^64
class Sh:
    """Arithmetic 2-of-2 sharing: value = a + b mod 2^64 (a at P0, b at P1)."""
    __slots__ = ("a", "b")

    def __init__(self, a, b):
        self.a, self.b = np.asarray(a, U64), np.asarray(b, U64)

    @staticmethod
    def public(x):
        x = enc(x)
        return Sh(x, np.zeros_like(x))

    def __add__(self, o):
        if isinstance(o, Sh):
            return Sh(self.a + o.a, self.b + o.b)
        a = self.a + enc(o)
        return Sh(a, self.b + np.zeros_like(a))          # keep both shares the same shape

    __radd__ = __add__

    def __sub__(self, o):
        return self + (0 - o if isinstance(o, Sh) else -np.asarray(o, np.int64))

    def __rsub__(self, o):                      # public - shared
        a = enc(o) - self.a
        return Sh(a, (U64(0) - self.b) + np.zeros_like(a))

    def __mul__(self, k):                       # by PUBLIC constant only (local)
        k = enc(k)
        return Sh(self.a * k, self.b * k)

    __rmul__ = __mul__

    def __getitem__(self, i):
        return Sh(self.a[i], self.b[i])

    def __setitem__(self, i, v):
        self.a[i], self.b[i] = v.a, v.b

    @property
    def shape(self):
        return self.a.shape

    def sum(self, axis=None):
        return Sh(self.a.sum(axis=axis, dtype=U64), self.b.sum(axis=axis, dtype=U64))

    def reshape(self, *s):
        return Sh(self.a.reshape(*s), self.b.reshape(*s))

    def matmul_public(self, M):                 # shared @ public int matrix (local)
        M = enc(M)
        return Sh(self.a @ M, self.b @ M)

    @staticmethod
    def cat(xs, axis=0):
        return Sh(np.concatenate([x.a for x in xs], axis), np.concatenate([x.b for x in xs], axis))

    @staticmethod
    def stack(xs, axis=0):
        return Sh(np.stack([x.a for x in xs], axis), np.stack([x.b for x in xs], axis))


class MPC:
    """P0/P1 compute on shares; helper = PRG-seeded dealer of correlated randomness.

    ponytail: Philox keyed from `secrets`, not a vetted CSPRNG (AES-CTR in deployment).
    """

    def __init__(self, pcg=False):
        """pcg=False: helper client deals triples (model A).  pcg=True: the two computing parties
        expand triples themselves from silent-OT/VOLE PCG seeds (model C, no helper); simulated,
        counted as correlations, not bytes (PCG expansion is local after a sublinear setup)."""
        self.g = np.random.Generator(np.random.Philox(key=secrets.randbits(128)))
        self.st = defaultdict(int)
        self.pcg = pcg

    def _corr(self, n, kind):
        if self.pcg:
            self.st[f"pcg_{kind}"] += n
        else:
            self.st["bytes_helper"] += 8 * n

    def rand(self, shape):
        n = int(np.prod(shape)) if len(shape) else 1
        return self.g.bit_generator.random_raw(n).astype(U64).reshape(shape)

    def split(self, x):
        x = np.asarray(x, U64)
        r = self.rand(x.shape)
        return Sh(r, x - r)

    def share(self, x):                          # owner sends one share to each compute party
        self.st["bytes_input"] += 16 * np.size(x)
        return self.split(x)

    def open(self, x):                           # both compute parties learn x (masked values only)
        self.st["bytes_open"] += 16 * x.a.size
        return x.a + x.b

    # ---- arithmetic multiplication (Beaver)
    def mul(self, x, y):
        xa, ya = np.broadcast_arrays(x.a, y.a)
        xb, yb = np.broadcast_arrays(x.b, y.b)
        x, y = Sh(xa, xb), Sh(ya, yb)
        a, b = self.rand(x.shape), self.rand(x.shape)
        A, B, C = self.split(a), self.split(b), self.split(a * b)   # helper -> P0, P1
        self._corr(a.size, "mul_triples")                           # dealer: seed-compressed C share
        d, e = self.open(x - A), self.open(y - B)
        self.st["mults"] += a.size
        return Sh(C.a + d * B.a + e * A.a + d * e, C.b + d * B.b + e * A.b)

    # ---- boolean AND on XOR-shared words
    def and_(self, x, y):
        a, b = self.rand(x[0].shape), self.rand(x[0].shape)
        c = a & b
        a0, b0, c0 = self.rand(a.shape), self.rand(a.shape), self.rand(a.shape)
        a1, b1, c1 = a ^ a0, b ^ b0, c ^ c0
        d = x[0] ^ a0 ^ x[1] ^ a1
        e = y[0] ^ b0 ^ y[1] ^ b1
        self.st["bytes_open"] += 32 * a.size
        self._corr(a.size, "and_triples")
        self.st["and_words"] += a.size
        return (c0 ^ (d & b0) ^ (e & a0) ^ (d & e), c1 ^ (d & b1) ^ (e & a1))

    # ---- boolean bit -> arithmetic bit
    def b2a(self, bits):
        r = self.rand(bits[0].shape) & ONE
        r0 = self.rand(r.shape) & ONE
        ra = self.split(r)
        self._corr(r.size, "dabits")
        z = bits[0] ^ r0 ^ bits[1] ^ r ^ r0
        self.st["bytes_open"] += 2 * r.size
        f = ONE - U64(2) * z
        return Sh(z + ra.a * f, ra.b * f)

    def ltz(self, x):
        """[x < 0] for |x| < 2^63: MSB via Kogge-Stone carry of the two 63-bit share halves."""
        self.st["ltz"] += x.a.size
        m = U64((1 << 63) - 1)
        lo0, lo1 = x.a & m, x.b & m
        z = np.zeros_like(lo0)
        g = self.and_((lo0, z), (z, lo1))                  # generate
        p = (lo0, lo1)                                     # propagate = lo0 ^ lo1
        for k in (1, 2, 4, 8, 16, 32):
            s = U64(k)
            t = self.and_(p, (g[0] << s, g[1] << s))
            g = (g[0] ^ t[0], g[1] ^ t[1])                  # generate/propagate disjoint: OR = XOR
            if k < 32:
                p = self.and_(p, (p[0] << s, p[1] << s))
        c0, c1 = (g[0] >> U64(62)) & ONE, (g[1] >> U64(62)) & ONE
        return self.b2a(((x.a >> U64(63)) ^ c0, (x.b >> U64(63)) ^ c1))

    def ge0(self, x):
        return 1 - self.ltz(x)

    def max(self, x, y):
        return x + self.mul(self.ltz(x - y), y - x)


# =============================================================== Phase 2: fuzzy circuit-PSI (BFV)
class BFV:
    """Public BFV parameters (shared) + one party's key pair.  SEAL via tenseal.sealapi."""
    _ctx = None

    @classmethod
    def context(cls):
        if cls._ctx is None:
            import tenseal.sealapi as sa
            parms = sa.EncryptionParameters(sa.SCHEME_TYPE.BFV)
            parms.set_poly_modulus_degree(POLY)
            parms.set_coeff_modulus(sa.CoeffModulus.BFVDefault(POLY, sa.SEC_LEVEL_TYPE.TC128))
            parms.set_plain_modulus(T)
            cls.sa, cls._ctx = sa, sa.SEALContext(parms, True, sa.SEC_LEVEL_TYPE.TC128)
            cls.ev, cls.be = sa.Evaluator(cls._ctx), sa.BatchEncoder(cls._ctx)
        return cls._ctx

    def __init__(self):
        ctx, sa = self.context(), self.sa
        kg = sa.KeyGenerator(ctx)
        self.pk = sa.PublicKey()
        kg.create_public_key(self.pk)
        self.decryptor = sa.Decryptor(ctx, kg.secret_key())

    @classmethod
    def encode(cls, v):
        p = cls.sa.Plaintext()
        cls.be.encode(np.asarray(v, np.int64).tolist(), p)
        return p

    @classmethod
    def encryptor(cls, pk):                      # anyone holding the public key
        return cls.sa.Encryptor(cls.context(), pk)

    def decrypt(self, ct):
        p = self.sa.Plaintext()
        self.decryptor.decrypt(ct, p)
        return np.asarray(self.be.decode_int64(p), np.int64)


SWITCH = 2                  # reply modulus switches: 432 KB -> 214 KB/ciphertext, ~48 bits noise budget left
_SIZE = {}


def _ct_bytes(ct, key):
    """serialized size of a ciphertext at a given level (measured once)."""
    if key not in _SIZE:
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ct.save(os.path.join(d, "c"))
            _SIZE[key] = os.path.getsize(os.path.join(d, "c"))
    return _SIZE[key]


def psi_shares(sigs, pairs, mpc, rng, st):
    """Packed fuzzy circuit-PSI: ALL signals of a pair in one message.

    Receiver j packs [z|s|v] of its n_j classes densely: slot k*n_j + b = y_b[k] (2 ciphertexts
    for 10 classes x 1103 dims).  For each own row a, sender i multiplies by x_a[k] per slot,
    adds a fresh uniform mask rho per slot, re-randomizes and modulus-switches.  j decrypts
    x_a[k]*y_b[k] + rho (uniform).  Both sides SUM their per-slot shares over k within each
    signal locally -> additive shares mod t of <x_a, y_b> per signal; no rotations/Galois keys.
    -> {signal: {(i,j): Sh C[ni,nj]}},  C = QSCALE^2 * cos.
    """
    keys = list(sigs)
    ids = sorted({i for p in pairs for i in p})
    X = {i: np.concatenate([np.rint(sigs[k][i] * QSCALE) for k in keys], 1).astype(np.int64) for i in ids}
    cut = np.cumsum([0] + [sigs[k][ids[0]].shape[1] for k in keys])
    by_j = defaultdict(list)
    for i, j in pairs:
        by_j[j].append(i)
    out = {k: {} for k in keys}
    BFV.context()
    ev = BFV.ev
    for j, senders in by_j.items():
        key = BFV()
        sa = key.sa
        nj, d = X[j].shape
        flat = X[j].T.reshape(-1)                                   # slot k*nj + b
        nct = -(-len(flat) // POLY)
        flat = np.r_[flat, np.zeros(nct * POLY - len(flat), np.int64)]
        enc_j = BFV.encryptor(key.pk)
        cts = []
        for c in range(nct):
            ct = sa.Ciphertext()
            enc_j.encrypt(BFV.encode(flat[c * POLY:(c + 1) * POLY]), ct)
            cts.append(ct)
        for i in senders:
            # ---- receiver -> sender: public key + nct ciphertexts (point-to-point)
            st["he_bytes"] += 541508 + nct * _ct_bytes(cts[0], 0)
            enc_i = BFV.encryptor(key.pk)                           # sender side: public key only
            ni = len(X[i])
            Ys = np.zeros((ni, nct * POLY), np.int64)
            Xs = np.zeros((ni, nct * POLY), np.int64)
            for a in range(ni):
                row = np.r_[np.repeat(X[i][a], nj), np.zeros(nct * POLY - nj * d, np.int64)]
                for c in range(nct):
                    sl = slice(c * POLY, (c + 1) * POLY)
                    rho = rng.integers(0, T, POLY)
                    r, z = sa.Ciphertext(), sa.Ciphertext()
                    ev.multiply_plain(cts[c], BFV.encode(row[sl]), r)
                    ev.add_plain_inplace(r, BFV.encode(rho))
                    enc_i.encrypt_zero(z)                           # re-randomize
                    ev.add_inplace(r, z)
                    for _ in range(SWITCH):
                        ev.mod_switch_to_next_inplace(r)
                    st["he_bytes"] += _ct_bytes(r, SWITCH)          # sender -> receiver
                    Ys[a, sl] = key.decrypt(r) % T                  # j's per-slot share
                    Xs[a, sl] = (-rho) % T                          # i's per-slot share
            # ---- local: sum per-slot shares over coordinates of each signal
            Ys = Ys[:, :nj * d].reshape(ni, d, nj)
            Xs = Xs[:, :nj * d].reshape(ni, d, nj)
            for k, lo, hi in zip(keys, cut[:-1], cut[1:]):
                xs, ys = Xs[:, lo:hi].sum(1) % T, Ys[:, lo:hi].sum(1) % T
                s = mpc.share(enc(xs)) + mpc.share(enc(ys))                         # s in [0, 2t)
                w = mpc.ge0(Sh.stack([s - (T + 1) // 2, s - (3 * T + 1) // 2]))   # [s>=t/2], [s>=3t/2]
                out[k][(i, j)] = s - w[0] * T - w[1] * T                           # centered
    return out


# =============================================================== Phase 2 (model C): VOLE inner products
def vole_shares(sigs, pairs, st, rng):
    """Derandomized VOLE: shares over Z_2^64 of <x_a, y_b> for every class pair, all signals at once.

    Random correlation (from a PCG, simulated): i gets U'[a,k], W'[a,k,b]; j gets D'[k,b], V'[a,k,b]
    with W' = U' * D' + V'.  i sends X - U' (n_i*d), j sends Y - D' (d*n_j); both uniformly masked.
    i: sum_k W' + U' * (Y - D')      j: sum_k (X - U') * Y - V'      ->  sum = X @ Y   (exact, no wrap)
    Cost per pair: 8 * d * (n_i + n_j) bytes + PCG setup (sublinear, not counted).
    -> {signal: {(i,j): Sh C[ni,nj]}}  (share a at client i, b at client j),  C = QSCALE^2 * cos.
    """
    keys = list(sigs)
    ids = sorted({i for p in pairs for i in p})
    Q = {i: enc(np.concatenate([np.rint(sigs[k][i] * QSCALE) for k in keys], 1).astype(np.int64)) for i in ids}
    cut = np.cumsum([0] + [sigs[k][ids[0]].shape[1] for k in keys])

    def R(*shape):
        return rng.bit_generator.random_raw(int(np.prod(shape))).astype(U64).reshape(shape)

    out = {k: {} for k in keys}
    for i, j in pairs:
        X, Y = Q[i], Q[j].T                                         # [ni, d], [d, nj]
        ni, d = X.shape
        nj = Y.shape[1]
        U_, D_, V_ = R(ni, d), R(d, nj), R(ni, d, nj)
        W_ = U_[:, :, None] * D_[None] + V_                          # PCG output (simulated)
        mx, my = X - U_, Y - D_                                      # the only messages
        st["vole_bytes"] += 8 * (mx.size + my.size)
        si = W_ + U_[:, :, None] * my[None]                          # at i
        sj = mx[:, :, None] * Y[None] - V_                           # at j
        for k, lo, hi in zip(keys, cut[:-1], cut[1:]):
            out[k][(i, j)] = Sh(si[:, lo:hi].sum(1, dtype=U64), sj[:, lo:hi].sum(1, dtype=U64))
    return out


def reshare(mpc, x):
    """i and j move their 2-of-2 sharing to P0/P1 without opening: each splits its own share."""
    r, s = mpc.rand(x.shape), mpc.rand(x.shape)
    mpc.st["bytes_reshare"] += 32 * x.a.size
    return Sh(r + s, (x.a - r) + (x.b - s))


# =============================================================== Phase 3: matching circuit
def ladder_bits(mpc, C, ladder):
    """shares of [cos > d] for d ascending, shape (m, *C.shape)."""
    d = np.sort(np.asarray(ladder, float))
    thr = np.floor(d * QSCALE ** 2).astype(np.int64).reshape((-1,) + (1,) * len(C.shape))
    return mpc.ltz(Sh.public(np.broadcast_to(thr, (len(d),) + C.shape)) - Sh.stack([C] * len(d)))


def bounds(bits, ladder):
    """rank_bounds on shares (exact, FIX units): lower d_R (-1 if R=0), upper d_{R+1} (1 if R=m)."""
    D = np.r_[-FIX, np.rint(np.sort(np.asarray(ladder, float)) * FIX).astype(np.int64), FIX]
    shp = (-1,) + (1,) * (len(bits.shape) - 1)
    lo = (bits * (D[1:-1] - D[:-2]).reshape(shp)).sum(0) + int(D[0])
    hi = (bits * (D[2:] - D[1:-1]).reshape(shp)).sum(0) + int(D[1])
    return lo, hi


def excl_max(mpc, V, axis):
    """max over the other entries of the same row (axis=1) / column (axis=0); -1 if none."""
    V = V if axis == 1 else Sh(V.a.T, V.b.T)
    n, m = V.shape
    none = Sh.public(np.full(n, -FIX))
    pre, suf = [none], [none]
    for k in range(1, m):
        pre.append(mpc.max(pre[-1], V[:, k - 1]))
        suf.append(mpc.max(suf[-1], V[:, m - k]))
    out = Sh.stack([mpc.max(pre[k], suf[m - 1 - k]) for k in range(m)], axis=1)
    return out if axis == 1 else Sh(out.a.T, out.b.T)


def one_sided(mpc, S):
    """row-wise: [S == rowmax] & [rowmax > 0] & [unique]  (S >= 0)."""
    n, m = S.shape
    mx = S[:, 0]
    for k in range(1, m):
        mx = mpc.max(mx, S[:, k])
    mxb = Sh(np.repeat(mx.a[:, None], m, 1), np.repeat(mx.b[:, None], m, 1))
    eq = mpc.ge0(S - mxb)                                        # S <= max always
    pos_unique = mpc.mul(mpc.ltz(0 - mx), mpc.ltz(eq.sum(1) - 2))
    return mpc.mul(eq, Sh(np.repeat(pos_unique.a[:, None], m, 1), np.repeat(pos_unique.b[:, None], m, 1)))


def precision_circuit(mpc, Ct, Ca, Cv, lt, la, lv, floor, margin):
    """verified_candidates + Mutual on shares for one client pair -> (support, edge, score)."""
    bt, ba, bv = ladder_bits(mpc, Ct, lt), ladder_bits(mpc, Ca, la), ladder_bits(mpc, Cv, lv)
    tl, _ = bounds(bt, lt)
    il, _ = bounds(ba, la)
    vl, vu = bounds(bv, lv)
    st_, si_ = bt[len(lt) - 1], ba[len(la) - 1]                  # strong = top rung cleared
    strong = st_ + si_ - mpc.mul(st_, si_)
    elig = mpc.mul(mpc.mul(bt[0], ba[0]), mpc.mul(strong, mpc.ge0(vl - int(round(floor * FIX)))))
    comp = mpc.max(excl_max(mpc, vu, 1), excl_max(mpc, vu, 0))
    mg = int(round(margin * FIX))
    ok_t, ok_i = mpc.ge0(vl - comp + mg), mpc.ge0(vl - comp - mg)
    supported = mpc.mul(elig, ok_i + mpc.mul(st_, ok_t - ok_i))
    score = mpc.mul(supported, tl + il + vl)                     # 3x plaintext score (same order)
    row = one_sided(mpc, score)
    col = one_sided(mpc, Sh(score.a.T, score.b.T))
    edge = mpc.mul(row, Sh(col.a.T, col.b.T))
    return supported, edge, score


# =============================================================== Phase 3: grouping circuit
def bitonic_desc(mpc, F):
    """Oblivious sort of rows of F (shared, col 0 = key) by key, descending."""
    n = F.shape[0]
    for k in (2 ** e for e in range(1, n.bit_length())):
        j = k // 2
        while j:
            i = np.arange(n)
            l_ = i ^ j
            sel = l_ > i
            lo, hi = i[sel], l_[sel]
            up = (lo & k) == 0                                   # up block: larger first
            diff = F[hi][:, 0] - F[lo][:, 0]
            s = mpc.ltz(Sh(np.where(up, U64(0) - diff.a, diff.a), np.where(up, U64(0) - diff.b, diff.b)))
            # up: swap if key_lo < key_hi  (ltz(lo-hi)); down: swap if key_hi < key_lo
            dF = mpc.mul(Sh(s.a[:, None], s.b[:, None]), F[hi] - F[lo])
            F[lo], F[hi] = F[lo] + dF, F[hi] - dF
            j //= 2
    return F


def grouping_circuit(mpc, U, V, E, W, N, client_of, H, B):
    """Weight-ordered union-find on shares (== rp.global_table). -> shared root label per node.
    U, V: public endpoint node ids of candidate (i,a)-(j,b), in plaintext edge order; E, W: shared."""
    M = len(U)
    P2 = 1 << max(1, (M - 1).bit_length())
    F = Sh.public(np.zeros((P2, 4), np.int64))                   # [key, edge, u, v]
    F[:, 0] = Sh.public(np.full(P2, -1))                         # padding sorts last
    F[:M, 0] = mpc.mul(E, W * M + (M - 1 - np.arange(M)))        # desc weight, then input order
    F[:M, 1] = E
    F.a[:M, 2], F.a[:M, 3] = enc(U), enc(V)                      # endpoints travel as indices
    F = bitonic_desc(mpc, F)
    K = max(client_of) + 1
    Cm = np.zeros((N, K), np.int64)
    Cm[np.arange(N), client_of] = 1
    ar = np.arange(N)
    eq = lambda X: 1 - mpc.ltz(X) - mpc.ltz(0 - X)               # [x == 0]
    col = lambda x: Sh(x.a[:, None], x.b[:, None])
    lab = Sh.public(ar)                                          # label = root node of component
    Hbar = 1 - H
    for s in range(B):                                           # all real edges lie in the top B
        e = F[s, 1]
        o = eq(Sh.stack([F[s, 2] - ar, F[s, 3] - ar]))           # one-hot of u, v
        AB = mpc.mul(o, Sh.stack([lab, lab])).sum(1)             # labels A = lab[u], B = lab[v]
        inAB = eq(Sh.stack([lab - AB[0], lab - AB[1]]))          # members of comp(u), comp(v)
        inA, inB = inAB[0], inAB[1]
        # same component => shares a client => conflict; so no separate A==B test
        conflict = mpc.mul(inA.matmul_public(Cm), inB.matmul_public(Cm)).sum()
        viol = mpc.mul(inA, mpc.mul(Hbar, Sh(inB.a[None], inB.b[None])).sum(1)).sum()
        z = mpc.ge0(Sh.stack([0 - conflict, 0 - viol]))          # [conflict == 0], [viol == 0]
        ok = mpc.mul(e, mpc.mul(z[0], z[1]))
        lab = lab + mpc.mul(mpc.mul(ok, inB), AB[0] - AB[1])     # relabel comp(v) -> A
    return lab


# =============================================================== full protocol
def run_circuit(clients, P, ladder=rp.LADDER, tau=None, coord="log_whiten", min_samples=10, seed=0,
                K_text=None, verify_ladder=rp.VERIFY_LADDER, verify_floor=.90, verify_margin=.01,
                log=print, masked_out=None, model="A", **_):
    """model "A": BFV PSI -> all matching + grouping by P0/P1 with a helper client dealing triples.
    model "C": VOLE PSI -> each pair (i,j) runs ITS OWN matching as 2PC on the shares it holds ->
    reshare edge/weight/support to P0/P1 -> grouping 2PC.  Triples from silent-OT/VOLE PCGs: no helper."""
    rng = np.random.default_rng(seed)                            # masked ids: same as rt_protocol
    lad = lambda k: np.asarray(ladder[k] if isinstance(ladder, dict) else ladder, dtype=float)
    tau = 0.05 if tau is None else tau
    K_text = None if K_text is None else rp.unit(np.asarray(K_text, float))
    t0 = time.time()

    # ---------------- Phase 1 (local; identical to rt_protocol attn_filter/precision)
    inv, sigs, n_elig = {}, defaultdict(dict), {}
    for i, c in clients.items():
        elig = [a for a in sorted(c["summ"]) if c["count"][a] >= min_samples]
        rest = [a for a in sorted(c["count"]) if a not in set(elig)]
        elig = [elig[k] for k in rng.permutation(len(elig))]
        inv[i] = {m: a for m, a in enumerate(elig + rest)}
        n_elig[i] = len(elig)
        if not elig:
            continue
        src = c.get("precision_summ", c["summ"])
        Rq = np.stack([src[a] for a in elig])
        sigs["verify"][i] = rp.unit(Rq)
        sigs["aff"][i] = rp.center_unit(rp.aff_profile(Rq, P))
        Ki = rp.unit(np.asarray(c["anchor_vecs"], float)) if "anchor_vecs" in c else K_text
        alpha, pooled, _ = rp.keyword_attn(c, elig, Ki, tau)
        sigs["main"][i] = (rp.anchor_log_coordinates(alpha, Ki) if coord == "log_whiten"
                           else rp.center_unit(alpha if coord == "alpha" else pooled))
    ids = sorted(i for i in clients if n_elig[i] > 0)
    pairs = [(i, j) for x, i in enumerate(ids) for j in ids[x + 1:]]
    t1 = time.time()

    # ---------------- Phase 2: pairwise fuzzy circuit-PSI (clients only)
    vole = model == "C"
    mpc, st = MPC(pcg=vole), defaultdict(int)                    # model C: grouping 2PC of P0/P1
    pair_mpc = MPC(pcg=True) if vole else mpc                    # model C: per-pair 2PC of i, j
    he_rng = np.random.default_rng(secrets.randbits(128))
    three = {k: sigs[k] for k in ("main", "aff", "verify")}
    C = vole_shares(three, pairs, st, he_rng) if vole else psi_shares(three, pairs, mpc, he_rng, st)
    t2 = time.time()

    # ---------------- Phase 3: matching circuit (per pair) + grouping circuit (P0, P1, helper)
    base, off = {}, 0
    for i in ids:
        base[i], off = off, off + n_elig[i]
    N = off
    client_of = np.array([k for k, i in enumerate(ids) for _ in range(n_elig[i])])
    H = Sh.public(np.zeros((N, N), np.int64))
    U, V, E, W = [], [], [], []
    for i, j in pairs:
        sup, edge, score = precision_circuit(pair_mpc, C["main"][i, j], C["aff"][i, j], C["verify"][i, j],
                                             lad("main"), lad("aff"), np.asarray(verify_ladder, float),
                                             verify_floor, verify_margin)
        if vole:                                                 # i, j -> P0, P1 (no opening)
            sup, edge, score = reshare(mpc, sup), reshare(mpc, edge), reshare(mpc, score)
        ri, rj = slice(base[i], base[i] + n_elig[i]), slice(base[j], base[j] + n_elig[j])
        H.a[ri, rj], H.b[ri, rj] = sup.a, sup.b
        H.a[rj, ri], H.b[rj, ri] = sup.a.T, sup.b.T
        a, b = np.divmod(np.arange(n_elig[i] * n_elig[j]), n_elig[j])
        U.append(base[i] + a), V.append(base[j] + b)
        E.append(edge.reshape(-1)), W.append(score.reshape(-1))
    B = sum(min(n_elig[i], n_elig[j]) for i, j in pairs)         # public bound on #Mutual edges
    t3 = time.time()
    label = (grouping_circuit(mpc, np.concatenate(U), np.concatenate(V), Sh.cat(E), Sh.cat(W),
                              N, client_of, H, B) if pairs else Sh.public(np.arange(N)))
    t4 = time.time()

    # ---------------- Output: P0, P1 send label shares to the server, which reconstructs
    mpc.st["bytes_to_server"] += 16 * N
    lab = dec(label.a + label.b)
    node = [(i, m) for i in ids for m in range(n_elig[i])]
    root = {(i, m): (i, m) for i in clients for m in inv[i]}
    root.update({node[n]: node[int(lab[n])] for n in range(N)})
    gid = {r: g for g, r in enumerate(sorted(set(root.values())))}
    table = {k: gid[r] for k, r in root.items()}

    if masked_out is not None:
        masked_out.update({(i, a): m for i in inv for m, a in inv[i].items()})
    out = {(i, inv[i][m]): g for (i, m), g in table.items()}
    if vole:
        for k, v in pair_mpc.st.items():
            st[f"pair2pc_{k}"] += v
    diag = {**{k: np.array(v) for k, v in st.items()}, **{k: np.array(v) for k, v in mpc.st.items()},
            "n_nodes": np.array(N), "n_candidates": np.array(sum(map(len, U))), "union_steps": np.array(B),
            "sec_phase1": np.array(t1 - t0), "sec_psi": np.array(t2 - t1),
            "sec_match": np.array(t3 - t2), "sec_group": np.array(t4 - t3)}
    psi_b = st["vole_bytes"] + st["he_bytes"]
    mpc_b = (mpc.st["bytes_open"] + mpc.st["bytes_helper"] + mpc.st["bytes_reshare"]
             + st["pair2pc_bytes_open"])
    log(f"[RT] circuit-PSI model {model}: {N} classes, {len(pairs)} pairs, {len(set(table.values()))} global ids"
        f" | PSI {psi_b / 1e6:.1f} MB, MPC {mpc_b / 1e6:.0f} MB | psi {t2 - t1:.0f}s match {t3 - t2:.0f}s"
        f" group {t4 - t3:.0f}s")
    return out, [], diag                                        # edges are never revealed


def run_rt_protocol(clients, P, method="filter", psi="he", **kw):
    """Drop-in for rt_protocol.run_rt_protocol.
    psi="circuit": BFV circuit-PSI, helper-dealt 2PC (model A).
    psi="vole":    VOLE circuit-PSI, per-pair 2PC + resharing, PCG triples, no helper (model C)."""
    if psi not in ("circuit", "vole"):
        return rp.run_rt_protocol(clients, P, method=method, psi=psi, **kw)
    if method != "attn_filter" or kw.get("attn_match", "precision") != "precision":
        raise ValueError(f"psi={psi} implements method=attn_filter, attn_match=precision")
    with np.errstate(over="ignore"):          # Z_2^64 wraparound is intended
        return run_circuit(clients, P, model="C" if psi == "vole" else "A", **kw)
