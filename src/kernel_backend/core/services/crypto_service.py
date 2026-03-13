from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
    load_pem_public_key,
)

import rfc8785

from kernel_backend.core.domain.manifest import CryptographicManifest
from kernel_backend.core.domain.watermark import WatermarkID


def generate_keypair() -> tuple[str, str]:
    """Returns (private_key_pem, public_key_pem) as UTF-8 strings."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    private_pem = private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    ).decode("utf-8")
    public_pem = public_key.public_bytes(
        encoding=Encoding.PEM,
        format=PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")
    return private_pem, public_pem


def sign_manifest(manifest: CryptographicManifest, private_key_pem: str) -> bytes:
    """
    Serialize manifest with rfc8785.dumps(), then sign with Ed25519.
    Returns 64-byte raw signature.

    CRITICAL: Ed25519PrivateKey.sign(data) takes raw bytes and hashes
    internally per RFC 8032. Do NOT pre-hash with SHA-256.
    CRITICAL: Use rfc8785.dumps() — never json.dumps(sort_keys=True).
    """
    canonical = rfc8785.dumps(_manifest_to_dict(manifest))
    private_key = load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    return private_key.sign(canonical)  # type: ignore[union-attr]


def verify_manifest(
    manifest: CryptographicManifest, signature: bytes, public_key_pem: str
) -> bool:
    """
    Returns True if valid. Returns False on InvalidSignature — never raises.
    """
    canonical = rfc8785.dumps(_manifest_to_dict(manifest))
    public_key = load_pem_public_key(public_key_pem.encode("utf-8"))
    try:
        public_key.verify(signature, canonical)  # type: ignore[union-attr]
        return True
    except InvalidSignature:
        return False


def derive_wid(signature: bytes, content_id: str) -> WatermarkID:
    """
    HKDF-SHA256:
      ikm    = signature (64-byte Ed25519 signature)
      salt   = content_id.encode("utf-8")
      info   = b"kernel_wid_v2"
      length = 16
    Returns WatermarkID(data=<16 bytes>).
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=16,
        salt=content_id.encode("utf-8"),
        info=b"kernel_wid_v2",
    )
    return WatermarkID(data=hkdf.derive(signature))


def _manifest_to_dict(manifest: CryptographicManifest) -> dict:
    """Convert manifest to a JSON-serializable dict."""
    return {
        "author_id": manifest.author_id,
        "author_public_key": manifest.author_public_key,
        "content_hash_sha256": manifest.content_hash_sha256,
        "content_id": manifest.content_id,
        "created_at": manifest.created_at,
        "fingerprints_audio": manifest.fingerprints_audio,
        "fingerprints_video": manifest.fingerprints_video,
        "schema_version": manifest.schema_version,
    }
