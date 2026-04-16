# Kernel Security Backend -- Domain & Business Architecture

## Overview

Kernel Security is a **forensic digital watermarking platform** that embeds invisible, cryptographically-bound identity marks into audio and video media. It operates as a multi-tenant API serving three core business functions:

1. **Identity Management** -- Issue and govern cryptographic identities tied to organizations.
2. **Multimodal Signing Engine** -- Embed acoustic and visual watermarks that survive common transcoding and compression.
3. **Verification Flow** -- Authenticate the origin and integrity of any media file, even after degradation.

---

## 1. Identity Management

### Problem

Before any media can be signed, the platform must answer: *who is signing?*  
Kernel establishes identity through three layers: **organizations**, **memberships**, and **cryptographic certificates**.

### Organizations

An **Organization** is the top-level tenant. Each organization:

- Has a unique, secret **pepper** (32-byte random value) that seeds all watermark randomization. Two organizations that sign the same file produce entirely different watermark patterns.
- Issues **API keys** (prefixed `krnl_`) for programmatic access. Keys are stored as SHA-256 hashes; the plaintext is shown exactly once at creation.
- Manages **members** with two roles: `admin` (full control, invitation power) and `member` (sign and verify within the org).

### Invitations

Admins invite new members by email. The invitation flow:

1. Admin creates an invitation targeting an email address.
2. Platform generates a one-time token (UUID), valid for 7 days.
3. An email is sent with a link to accept the invitation.
4. The invitee authenticates, presents the token, and is added to the organization as a `member`.

Invitations are single-use; once accepted or expired they cannot be reused.

### Cryptographic Certificates

Each user generates an **Ed25519 keypair** that becomes their signing identity:

| Field | Purpose |
|-------|---------|
| `author_id` | Unique, platform-assigned identifier bound to the user account |
| `public_key_pem` | Ed25519 public key -- stored permanently, used for verification |
| `private_key_pem` | Ed25519 private key -- returned **once** at generation, never stored |
| `name` / `institution` | Human-readable metadata for attribution |

The private key never leaves the user's control after generation. All subsequent signing operations require the user to supply their private key. This means the platform itself **cannot forge signatures** on behalf of any user.

### Authentication Methods

The API supports three authentication mechanisms:

| Method | Use Case | Token Format |
|--------|----------|--------------|
| **API Key** | Server-to-server / CI pipelines | `krnl_<hex>` in `Authorization: Bearer` header |
| **Local Admin JWT** | Platform administration | HS256 JWT, 8-hour TTL |
| **Stack Auth Session** | Frontend users (OAuth) | RS256/ES256 JWT verified against Stack Auth JWKS |

---

## 2. Multimodal Signing Engine

### Problem

A signed file must carry an invisible, resilient proof of origin that:

- Survives re-encoding (H.264, AAC), resolution changes, and bitrate reduction.
- Cannot be forged or transplanted from one file to another.
- Can be extracted without the original (blind verification).

Kernel solves this by embedding **two complementary signal layers** into the media.

### Layer 1 -- Watermark Identity (WID)

The WID is a **128-bit cryptographic identifier** derived from the content's Ed25519 manifest signature via HKDF-SHA256. It is unique per (file, signer) pair and is the authoritative proof of origin.

**Audio channel (DSSS + DWT):**

The WID is spread across the audio spectrum using Direct Sequence Spread Spectrum modulation, embedded into the detail band of a Daubechies-4 wavelet decomposition. Each audio segment carries one Reed-Solomon symbol (8 bits). The embedding power is calibrated to survive 256 kbps AAC encoding while remaining inaudible.

**Video channel (QIM + DCT):**

The WID is quantized into the AC coefficients of 4x4 block DCTs on the luminance plane using Quantization Index Modulation. Block positions and coefficient selections are randomized per segment via HMAC-seeded hopping, making the watermark location unpredictable without knowledge of the organization's pepper. The quantization step is calibrated for survival under H.264 at CRF <= 28.

### Layer 2 -- Perceptual Fingerprint

The fingerprint is a **content-derived hash** used for fast candidate lookup during verification. It does not carry identity -- it answers *"which registered content does this file correspond to?"*

**Audio fingerprint:** Log-mel spectrogram segments (2-second windows, 50% overlap) are projected through a keyed DCT basis into 64-bit hashes. Speech-optimized frequency range (300-8000 Hz) with spectral subtraction for noise robustness.

**Video fingerprint:** Grayscale 32x32 thumbnails of 5-second segments are decomposed with 2D DCT and projected into a 64-bit keyed hash.

Fingerprints are stored in a registry and matched using Hamming distance (threshold: < 10 bits).

### Reed-Solomon Error Correction

The 128-bit WID (16 bytes) is expanded into *n* Reed-Solomon symbols (n in [17, 255]) distributed across media segments. This provides resilience against segment-level loss: even if substantial portions of the media are corrupted or truncated, the WID can be recovered as long as enough segments remain intact.

### Cryptographic Manifest

Every signed file produces a **CryptographicManifest** -- a canonical JSON document containing:

- `content_id`: Deterministic identifier derived from the signer's key.
- `author_id` and `public_key_pem`: Signer attribution.
- `content_hash_sha256`: Hash of the original file.
- `active_signals`: Which watermark types were embedded (audio WID, video WID, fingerprints).
- `rs_n`: Number of Reed-Solomon symbols used.
- `embedding_params`: Exact DSP parameters used for embedding.

The manifest is signed with Ed25519. Both the manifest JSON and its 64-byte signature are stored alongside the registry entry.

### Encode Throughput

The video encode stage (decode → embed → libx264) dominates wall-clock time on
long clips. The signing engine partitions the encode across N parallel chunk
workers when the source has enough segments; each worker processes a contiguous
span with a one-segment guard band on each side and the trimmed chunks are
re-joined with the ffmpeg concat demuxer. WID derivation, RS encoding, and the
per-frame embedding math are byte-identical to the sequential path — only the
encoder work is parallelized. Below the configured per-chunk segment floor the
planner collapses to a single chunk and the engine falls back to the sequential
encode. See `arch/ARCHITECTURE.md` for the chunked pipeline details.

### Organization-Scoped Isolation

Each organization's **pepper** seeds the PRNG chains used for:

- Fingerprint projection keys (different peppers produce different hash values for the same content).
- Hopping patterns (watermark placement in the frequency/spatial domain).

This means an organization cannot extract or verify watermarks belonging to another organization, even with full access to the media file.

---

## 3. Verification Flow

### Problem

Given an arbitrary media file -- potentially re-encoded, cropped, or partially corrupted -- determine:

1. Whether the file was signed on the platform.
2. Who signed it.
3. Whether the content has been tampered with.

### Two-Phase Verification Pipeline

**Phase A -- Candidate Identification (fast, approximate)**

Perceptual fingerprints are extracted from the submitted file and compared against the registry using Hamming distance. If a match is found (< 10 bits), the candidate's `content_id` and associated metadata are retrieved. This phase is O(registry_size) and handles format changes, bitrate shifts, and mild edits.

**Phase B -- Cryptographic Authentication (precise, authoritative)**

Using the candidate's stored parameters, the verifier:

1. Extracts WID symbols from audio and/or video segments.
2. Runs Reed-Solomon decoding (recovering from erasures).
3. Compares the extracted WID against the stored WID.
4. Verifies the Ed25519 signature of the cryptographic manifest.

A file is **VERIFIED** if and only if both conditions hold:

```
extracted_WID == stored_WID  AND  Ed25519_signature_valid
```

Any other outcome produces a **RED** verdict with a specific reason:

| Red Reason | Meaning |
|------------|---------|
| `CANDIDATE_NOT_FOUND` | No matching fingerprint in the registry |
| `WID_MISMATCH` | Watermark decoded but doesn't match -- possible tampering |
| `SIGNATURE_INVALID` | Manifest signature verification failed |
| `WID_UNDECODABLE` | Too many RS erasures -- severe degradation |
| `WATERMARK_DEGRADED` | Partial watermark, WID unrecoverable |

For audiovisual content, audio and video channels are verified independently and reported separately.

### Public vs. Authenticated Verification

| Endpoint | Auth Required | Pepper Strategy |
|----------|---------------|-----------------|
| `POST /verify` | Yes | Uses caller's organization pepper (fast) |
| `POST /verify/public` | No | Tries all organization peppers (slower, discovers origin org) |

Public verification enables third-party fact-checkers and platforms to authenticate content without holding credentials.

---

## Business Domain Summary

```
                        +----------------------------+
                        |    Organization Tenant      |
                        |  (pepper, API keys, roles)  |
                        +----------------------------+
                                    |
                     +--------------+--------------+
                     |                             |
              +------+------+            +---------+--------+
              |   Identity  |            |   Invitation     |
              |  Ed25519    |            |  Email + Token   |
              |  Keypair    |            |  7-day expiry    |
              +------+------+            +------------------+
                     |
         +-----------+-----------+
         |                       |
   +-----+------+        +------+------+
   |   Signing   |        | Verification|
   |   Engine    |        |   Engine    |
   +-----+------+        +------+------+
         |                       |
   +-----+------+        +------+------+
   | Audio WID   |        | Phase A:    |
   | (DSSS+DWT)  |        | Fingerprint |
   +-------------+        | Lookup      |
   | Video WID   |        +-------------+
   | (QIM+DCT)   |        | Phase B:    |
   +-------------+        | WID + Ed25519|
   | Fingerprints|        | Verification|
   +-------------+        +-------------+
   | Reed-Solomon|
   | Error Codes |
   +-------------+
```

---

## Key Design Decisions

1. **Private key never stored.** The platform cannot sign on behalf of users, providing non-repudiation.
2. **Organization peppers isolate tenants.** Watermark patterns are cryptographically unique per organization.
3. **Two-layer verification.** Fingerprints provide fast lookup; WID provides cryptographic proof. Neither alone is sufficient.
4. **Reed-Solomon redundancy.** Watermarks survive partial media corruption or truncation.
5. **Blind verification.** The original file is not needed -- verification works from the watermarked copy alone.
6. **Public verification endpoint.** Third parties can authenticate content without platform credentials.
