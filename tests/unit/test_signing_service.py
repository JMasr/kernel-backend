import numpy as np

from kernel_backend.core.services.signing_service import _PCMChunkScratch


def test_pcm_chunk_scratch_roundtrip():
    """Chunks written to scratch must come back bit-identical."""
    rng = np.random.default_rng(42)
    originals = [rng.standard_normal(88200).astype(np.float32) for _ in range(5)]

    with _PCMChunkScratch() as scratch:
        for c in originals:
            scratch.append(c)
        scratch.finalize()
        assert len(scratch) == 5
        roundtripped = list(scratch)

    assert len(roundtripped) == 5
    for a, b in zip(originals, roundtripped, strict=True):
        np.testing.assert_array_equal(a, b)


def test_pcm_chunk_scratch_variable_length_tail():
    """Last chunk is typically shorter than others — must round-trip correctly."""
    chunks = [
        np.full(88200, 0.25, dtype=np.float32),
        np.full(88200, -0.5, dtype=np.float32),
        np.full(12345, 0.1, dtype=np.float32),
    ]
    with _PCMChunkScratch() as scratch:
        for c in chunks:
            scratch.append(c)
        got = list(scratch)
    assert [a.size for a in got] == [88200, 88200, 12345]
    for a, b in zip(chunks, got, strict=True):
        np.testing.assert_array_equal(a, b)


def test_pcm_chunk_scratch_cleanup_on_exit():
    """Temp file must be deleted when the context manager exits normally."""
    with _PCMChunkScratch() as scratch:
        scratch.append(np.zeros(100, dtype=np.float32))
        path = scratch._path
        assert path.exists()
    assert not path.exists()


def test_pcm_chunk_scratch_cleanup_on_exception():
    """Temp file must be deleted even if the context body raises."""
    path = None
    try:
        with _PCMChunkScratch() as scratch:
            scratch.append(np.zeros(100, dtype=np.float32))
            path = scratch._path
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    assert path is not None
    assert not path.exists()


def test_pcm_chunk_scratch_coerces_non_float32():
    """float64 inputs must be cast to float32 when written."""
    src = np.arange(100, dtype=np.float64) * 0.01
    with _PCMChunkScratch() as scratch:
        scratch.append(src)
        got = list(scratch)
    assert got[0].dtype == np.float32
    np.testing.assert_allclose(got[0], src.astype(np.float32), atol=0)
