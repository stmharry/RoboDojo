from __future__ import annotations

import logging
import os

import numpy as np

from robodojo.sim.utils.save_file import VideoStreamWriter, format_video_saved_message

logger = logging.getLogger(__name__)


class VideoService:
    def _stream_vision(self, env_idx, frame):
        """Append this env's per-camera RGB frames to its ffmpeg streams.

        Only the vision ("color") data is recorded; writers are created
        lazily on the first frame (when the resolution is known) and write
        to temporary files until the episode outcome decides the name.
        """
        vision = frame.get("vision") if isinstance(frame, dict) else None
        if not vision:
            return
        writers = self.video_writers.setdefault(env_idx, {})
        fps = self.obs_manager.collect_freq
        for cam_key, cam_data in vision.items():
            color = cam_data.get("color") if isinstance(cam_data, dict) else None
            if color is None:
                continue
            color = np.ascontiguousarray(color)
            if color.ndim != 3 or color.shape[2] not in (3, 4):
                continue
            if cam_key not in writers:
                height, width, channels = color.shape
                tmp_path = os.path.join(self._stream_dir, f"env{env_idx}_{cam_key}.tmp.mp4")
                writers[cam_key] = VideoStreamWriter(tmp_path, height, width, channels, fps=fps)
            writers[cam_key].append(color)

    def _sweep_stream_dir(self):
        """Remove orphan temp videos left by a previous hard kill/crash."""
        stream_dir = getattr(self, "_stream_dir", None)
        if not stream_dir or not os.path.isdir(stream_dir):
            return
        for name in os.listdir(stream_dir):
            if name.endswith(".tmp.mp4"):
                try:
                    os.remove(os.path.join(stream_dir, name))
                except Exception:
                    pass

    def _abort_video_writers(self, env_idx_list=None):
        """Close and delete partial videos for the given (or all) envs."""
        if env_idx_list is None:
            env_idx_list = list(self.video_writers.keys())
        for env_idx in list(env_idx_list):
            writers = self.video_writers.pop(env_idx, {})
            for writer in writers.values():
                try:
                    writer.abort()
                except Exception:
                    pass

    def save_video(self, env_idx, video_path, tag):
        writers = self.video_writers.pop(env_idx, {})
        for cam_key, writer in writers.items():
            tmp_path = writer.out_path
            final_path = video_path.replace(".mp4", f"_{cam_key}_{tag}.mp4")
            try:
                writer.close(announce=False)
            except Exception as e:
                logger.warning("[EvalEnv] Failed to finalize video for env %s cam %s: %s", env_idx, cam_key, e)
                writer.abort()
                continue
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            os.replace(tmp_path, final_path)
            logger.info(
                "%s",
                format_video_saved_message(
                    final_path,
                    writer.n_frames,
                    writer.width,
                    writer.height,
                    writer.fps,
                ),
            )
