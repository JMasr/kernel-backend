from __future__ import annotations

from pathlib import Path

import cv2
import ffmpeg
import numpy as np

from kernel_backend.core.domain.media import MediaProfile
from kernel_backend.core.ports.media import MediaPort


class MediaService(MediaPort):
    """FFmpeg-backed media I/O."""

    def probe(self, path: Path) -> MediaProfile:
        """ffprobe → MediaProfile. Raises ValueError if no streams found."""
        try:
            info = ffmpeg.probe(str(path))
        except ffmpeg.Error as e:
            raise ValueError(f"ffprobe failed for {path}: {e.stderr.decode()}") from e

        streams = info.get("streams", [])
        if not streams:
            raise ValueError(f"No streams found in {path}")

        has_video = any(s["codec_type"] == "video" for s in streams)
        has_audio = any(s["codec_type"] == "audio" for s in streams)

        video_stream = next((s for s in streams if s["codec_type"] == "video"), None)
        audio_stream = next((s for s in streams if s["codec_type"] == "audio"), None)

        width = int(video_stream["width"]) if video_stream else 0
        height = int(video_stream["height"]) if video_stream else 0

        fps = 0.0
        if video_stream:
            r = video_stream.get("r_frame_rate", "0/1")
            num, den = (int(x) for x in r.split("/"))
            fps = num / den if den else 0.0

        duration_s = float(info["format"].get("duration", 0))

        sample_rate = 0
        if audio_stream:
            sample_rate = int(audio_stream.get("sample_rate", 44100))

        return MediaProfile(
            has_video=has_video,
            has_audio=has_audio,
            width=width,
            height=height,
            fps=fps,
            duration_s=duration_s,
            sample_rate=sample_rate,
        )

    def decode_audio_to_pcm(
        self,
        path: Path,
        target_sample_rate: int = 44100,
    ) -> tuple[np.ndarray, int]:
        """[DEPRECATED] Decode audio track → mono float32 PCM in [-1.0, 1.0]."""
        try:
            out, _ = (
                ffmpeg
                .input(str(path))
                .output("pipe:", format="s16le", ac=1, ar=target_sample_rate)
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise ValueError(
                f"FFmpeg decode failed for {path}: {e.stderr.decode()}"
            ) from e
        samples = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, target_sample_rate

    def iter_audio_segments(
        self,
        path: Path,
        segment_duration_s: float = 2.0,
        target_sample_rate: int = 44100,
    ):
        """
        Lazily yield (segment_idx, samples, sample_rate) for each audio segment.
        Reads from an FFmpeg subprocess pipe in chunks to prevent OOM on long files.
        """
        import subprocess

        bytes_per_sample = 2  # s16le = 2 bytes
        samples_per_segment = int(segment_duration_s * target_sample_rate)
        chunk_bytes = samples_per_segment * bytes_per_sample

        cmd = [
            "ffmpeg",
            "-i", str(path),
            "-f", "s16le",
            "-ac", "1",
            "-ar", str(target_sample_rate),
            "-loglevel", "quiet",
            "pipe:1"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        try:
            segment_idx = 0
            while True:
                # Read exactly chunk_bytes
                raw_data = process.stdout.read(chunk_bytes)
                if not raw_data:
                    break
                
                # Convert to float32
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                yield segment_idx, samples, target_sample_rate
                
                segment_idx += 1
        finally:
            if process.stdout:
                process.stdout.close()
            process.terminate()
            process.wait()

    def encode_audio_from_pcm(
        self,
        samples: np.ndarray,
        sample_rate: int,
        output_path: Path,
        codec: str = "aac",
        bitrate: str = "256k",
    ) -> None:
        """float32 PCM → encoded audio file via FFmpeg."""
        pcm = (samples * 32768.0).astype(np.int16)
        try:
            (
                ffmpeg
                .input("pipe:", format="s16le", ac=1, ar=sample_rate)
                .output(str(output_path), acodec=codec, **{"b:a": bitrate})
                .overwrite_output()
                .run(input=pcm.tobytes(), capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise ValueError(f"FFmpeg encode failed: {e.stderr.decode()}") from e

    def encode_audio_stream(
        self,
        sample_rate: int,
        output_path: Path,
        codec: str = "aac",
        bitrate: str = "256k",
    ):
        """
        Return a fast-running FFmpeg Popen encoding process.
        Write int16 raw PCM bytes to process.stdin.write(), then close().
        """
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le", "-ac", "1", "-ar", str(sample_rate),
            "-i", "pipe:0",
            "-acodec", codec, "-b:a", bitrate,
            str(output_path)
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def open_video_encode_stream(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: Path,
    ):
        """
        Opens an FFmpeg subprocess for streaming H.264 lossless video encode.
        Caller writes yuv.tobytes() per frame (yuvj420p full-range),
        then calls stdin.close() and wait().
        """
        import subprocess

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "yuvj420p",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-pix_fmt", "yuvj420p",
            "-loglevel", "quiet",
            str(output_path),
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def mux_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
    ) -> None:
        """Combine video stream (copy) + new audio into output container."""
        try:
            video_in = ffmpeg.input(str(video_path))
            audio_in = ffmpeg.input(str(audio_path))
            (
                ffmpeg
                .output(
                    video_in.video, audio_in.audio, str(output_path),
                    vcodec="copy", acodec="copy",
                )
                .overwrite_output()
                .run(capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise ValueError(f"FFmpeg mux failed: {e.stderr.decode()}") from e

    def read_video_frames(
        self,
        path: Path,
        start_frame: int = 0,
        n_frames: int | None = None,
    ) -> tuple[list[np.ndarray], float]:
        """Read BGR frames from video using OpenCV."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames = []
        count = 0
        while True:
            if n_frames is not None and count >= n_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            count += 1

        cap.release()
        return frames, fps

    def write_video_frames(
        self,
        frames: list[np.ndarray],
        fps: float,
        output_path: Path,
    ) -> None:
        """Write BGR frames to H.264 mp4 via a single ffmpeg pipe pass.

        Streams raw BGR frames directly to the H.264 encoder — no intermediate
        lossy codec (mp4v). A single-pass encode preserves QIM watermarks far
        better than the double-lossy mp4v → H.264 pipeline (Fix 7).
        """
        import subprocess

        if not frames:
            raise ValueError("No frames to write")

        h, w = frames[0].shape[:2]

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "yuvj420p",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-crf", "0", "-preset", "ultrafast",
            "-pix_fmt", "yuvj420p",
            "-loglevel", "quiet",
            str(output_path),
        ]

        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            for frame in frames:
                # Use YCrCb (full-range) matching embed_segment/extract_segment colour space.
                # yuvj420p = JPEG/full-range YUV420: Y full-res, Cb/Cr 2x subsampled.
                # cv2.COLOR_BGR2YCrCb layout: ch0=Y, ch1=Cr, ch2=Cb → FFmpeg order Y,U(Cb),V(Cr).
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                process.stdin.write(ycrcb[:, :, 0].tobytes())       # Y full res
                process.stdin.write(ycrcb[::2, ::2, 2].tobytes())   # U = Cb subsampled
                process.stdin.write(ycrcb[::2, ::2, 1].tobytes())   # V = Cr subsampled
            process.stdin.close()
            process.wait()
            if process.returncode != 0:
                raise ValueError(
                    f"FFmpeg video encode failed with return code {process.returncode}"
                )
        except Exception as e:
            process.kill()
            process.wait()
            raise ValueError(f"FFmpeg video encode failed: {e}") from e

    def seek_frame(self, path: Path, time_s: float) -> np.ndarray:
        """Read exactly one frame at the given timestamp via CAP_PROP_POS_MSEC."""
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot read frame at {time_s}s from {path}")
            return frame
        finally:
            cap.release()

    def iter_video_segments(
        self,
        path: Path,
        segment_duration_s: float = 5.0,
        frame_stride: int = 1,
    ):
        """
        Lazily yield (segment_idx, frames, fps) one segment at a time.

        frame_stride controls which frames are kept per segment:
        - frame_stride=1  → all frames (original behavior)
        - frame_stride=3  → every 3rd frame (~300 MB/segment at 1080p vs ~900 MB)

        No robustness loss with frame_stride=3 — extract_segment() uses majority
        vote across available frames regardless of count.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError(f"Invalid FPS ({fps}) for video: {path}")

            segment_frame_count = int(segment_duration_s * fps)
            if segment_frame_count <= 0:
                raise ValueError(
                    f"segment_duration_s={segment_duration_s} yields 0 frames at fps={fps}"
                )

            segment_idx = 0
            while True:
                frames: list[np.ndarray] = []
                frames_read = 0
                for i in range(segment_frame_count):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if i % frame_stride == 0:
                        frames.append(frame)
                    frames_read += 1

                if not frames:
                    break

                yield segment_idx, frames, fps

                if frames_read < segment_frame_count:
                    break

                segment_idx += 1
        finally:
            cap.release()
