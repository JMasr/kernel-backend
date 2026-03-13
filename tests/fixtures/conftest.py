import pytest
import pytest_asyncio
import numpy as np
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, async_sessionmaker
)
from kernel_backend.infrastructure.database.models import Base
from tests.fixtures.audio_signals import (
    white_noise, pink_noise, multitone_dwt,
    add_babble_noise, add_pink_noise,
    simulate_mp3_compression, simulate_voip_codec,
    SPEECH_RECORDINGS, MUSIC_RECORDINGS, ALL_LIBROSA_RECORDINGS,
)

SR = 44100


# ── Database fixture (unchanged from pre-Phase 2) ────────────────────────────

@pytest_asyncio.fixture
async def db_session():
    """
    In-memory SQLite session. Isolated per test.
    Use for repository tests — no Neon connection required.
    """
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    async with factory() as session:
        yield session
    await engine.dispose()


# ── Golden dataset fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def golden_speech(request) -> dict[str, np.ndarray]:
    """
    Session-scoped golden dataset. Loads all librosa speech recordings once.
    Returns dict: {recording_name: samples_float32}
    Tests that use this fixture are marked @pytest.mark.integration.
    """
    librosa = pytest.importorskip("librosa")
    dataset = {}
    for name in SPEECH_RECORDINGS:
        audio, _ = librosa.load(librosa.ex(name), sr=SR, mono=True)
        dataset[name] = audio.astype(np.float32)
    return dataset


@pytest.fixture(
    params=list(SPEECH_RECORDINGS.keys()),
    ids=list(SPEECH_RECORDINGS.keys()),
    scope="session",
)
def speech_sample(request) -> tuple[str, np.ndarray]:
    """
    Parametrized clean speech. Runs once per speech recording.
    Primary blocking fixture — if these fail, release is blocked.
    """
    librosa = pytest.importorskip("librosa")
    name = request.param
    audio, _ = librosa.load(librosa.ex(name), sr=SR, mono=True)
    return name, audio.astype(np.float32)


@pytest.fixture(
    params=list(ALL_LIBROSA_RECORDINGS.keys()),
    ids=list(ALL_LIBROSA_RECORDINGS.keys()),
    scope="session",
)
def librosa_signal(request) -> tuple[str, np.ndarray, int]:
    """All librosa recordings (speech + music). For broad coverage tests."""
    librosa = pytest.importorskip("librosa")
    name = request.param
    audio, sr = librosa.load(librosa.ex(name), sr=SR, mono=True)
    return name, audio.astype(np.float32), sr


# ── Synthetic baselines (informational, non-blocking) ────────────────────────

@pytest.fixture
def white_noise_signal() -> np.ndarray:
    return white_noise(5.0, SR, seed=42)

@pytest.fixture
def silence_signal() -> np.ndarray:
    from tests.fixtures.audio_signals import silence
    return silence(5.0, SR)


# ── Legacy synthetic fixtures (retained for DWT watermarking tests) ───────────

SYNTHETIC_SIGNAL_GENERATORS = [
    ("white_noise",   white_noise),
    ("pink_noise",    pink_noise),
    ("multitone_dwt", multitone_dwt),
]

@pytest.fixture(
    params=[name for name, _ in SYNTHETIC_SIGNAL_GENERATORS],
    ids=[name for name, _ in SYNTHETIC_SIGNAL_GENERATORS],
)
def synthetic_signal(request) -> tuple[str, np.ndarray]:
    """
    Parametrized fixture. Test runs 3×: white_noise, pink_noise, multitone_dwt.
    Returns (signal_name, samples_float32).
    Default: 5s at 44100 Hz.
    """
    name = request.param
    generator = dict(SYNTHETIC_SIGNAL_GENERATORS)[name]
    return name, generator(duration_s=5.0, sample_rate=SR)


@pytest.fixture(
    params=[name for name, _ in SYNTHETIC_SIGNAL_GENERATORS],
    ids=[name for name, _ in SYNTHETIC_SIGNAL_GENERATORS],
)
def audio_signal(request) -> tuple[str, object]:
    """
    Backwards-compatible fixture for DWT watermarking tests (pilot_tone, wid_beacon).
    Returns (signal_name, generator_callable(duration_s, sample_rate) -> ndarray).
    """
    name = request.param
    generator = dict(SYNTHETIC_SIGNAL_GENERATORS)[name]
    return name, generator
