import hashlib
import os
from datetime import datetime, timezone

from hypothesis import given, settings
import hypothesis.strategies as st

from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.watermark import WatermarkID
from kernel_backend.core.services.crypto_service import (
    derive_wid,
    generate_keypair,
    sign_manifest,
    streaming_file_hash,
    verify_manifest,
)


def _minimal_manifest() -> CryptographicManifest:
    return CryptographicManifest(
        content_id="test-content-001",
        content_hash_sha256="01" * 32,
        fingerprints_video=[],
        fingerprints_audio=["0102030405060708", "0a0b0c0d0e0f1011"],
        author_id="author-test-id",
        author_public_key="-----BEGIN PUBLIC KEY-----\ntest-key\n-----END PUBLIC KEY-----\n",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@given(
    st.text(min_size=1, max_size=100),
    st.text(min_size=1, max_size=100),
)
@settings(max_examples=20)
def test_sign_verify_roundtrip(name: str, institution: str) -> None:
    priv, pub = generate_keypair()
    manifest = _minimal_manifest()
    sig = sign_manifest(manifest, priv)
    assert len(sig) == 64
    assert verify_manifest(manifest, sig, pub) is True


def test_verify_wrong_keypair() -> None:
    priv1, pub1 = generate_keypair()
    priv2, pub2 = generate_keypair()
    sig = sign_manifest(_minimal_manifest(), priv1)
    assert verify_manifest(_minimal_manifest(), sig, pub2) is False


def test_verify_bit_flip() -> None:
    priv, pub = generate_keypair()
    sig = sign_manifest(_minimal_manifest(), priv)
    flipped = bytes([sig[0] ^ 0xFF]) + sig[1:]
    assert verify_manifest(_minimal_manifest(), flipped, pub) is False


def test_verify_wrong_manifest() -> None:
    priv, pub = generate_keypair()
    manifest = _minimal_manifest()
    sig = sign_manifest(manifest, priv)
    # Different content_id → different canonical bytes → signature invalid
    other = CryptographicManifest(
        content_id="different-content-id",
        content_hash_sha256=manifest.content_hash_sha256,
        fingerprints_video=manifest.fingerprints_video,
        fingerprints_audio=manifest.fingerprints_audio,
        author_id=manifest.author_id,
        author_public_key=manifest.author_public_key,
        created_at=manifest.created_at,
    )
    assert verify_manifest(other, sig, pub) is False


@given(
    st.binary(min_size=64, max_size=64),
    st.uuids(),
)
@settings(max_examples=30)
def test_derive_wid_deterministic(sig_bytes: bytes, content_uuid: object) -> None:
    wid1 = derive_wid(sig_bytes, str(content_uuid))
    wid2 = derive_wid(sig_bytes, str(content_uuid))
    assert wid1 == wid2
    assert len(wid1.data) == 16


@given(st.binary(min_size=64, max_size=64))
@settings(max_examples=30)
def test_derive_wid_different_content_ids(sig_bytes: bytes) -> None:
    wid1 = derive_wid(sig_bytes, "content-id-aaa")
    wid2 = derive_wid(sig_bytes, "content-id-bbb")
    assert wid1 != wid2


def test_derive_wid_returns_watermark_id() -> None:
    sig = b"\xAB" * 64
    wid = derive_wid(sig, "some-content-id")
    assert isinstance(wid, WatermarkID)
    assert len(wid.data) == 16


def test_generate_keypair_returns_pem_strings() -> None:
    priv, pub = generate_keypair()
    assert priv.startswith("-----BEGIN PRIVATE KEY-----")
    assert pub.startswith("-----BEGIN PUBLIC KEY-----")


def test_streaming_file_hash_matches_hashlib(tmp_path) -> None:
    """streaming_file_hash must be bit-for-bit identical to hashlib one-shot."""
    cases = [
        b"",
        b"x" * 17,
        b"y" * (2**16),
        b"z" * (2**16 + 12345),
    ]
    for i, data in enumerate(cases):
        p = tmp_path / f"case_{i}.bin"
        p.write_bytes(data)
        assert streaming_file_hash(p) == hashlib.sha256(data).hexdigest()


def test_streaming_file_hash_block_size_invariant(tmp_path) -> None:
    """Digest must be independent of block_size — validates the streaming contract."""
    p = tmp_path / "random.bin"
    p.write_bytes(os.urandom(1 << 20))  # 1 MiB

    expected = streaming_file_hash(p)
    for bs in (1, 17, 4096, 2**20, 2**21):
        assert streaming_file_hash(p, block_size=bs) == expected


def test_streaming_file_hash_algorithm_parameter(tmp_path) -> None:
    """algorithm=sha512 must call hashlib.new('sha512') correctly."""
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert (
        streaming_file_hash(p, algorithm="sha512")
        == hashlib.sha512(b"hello world").hexdigest()
    )
