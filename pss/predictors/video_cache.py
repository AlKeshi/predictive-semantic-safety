import copy
import hashlib
import json
import threading

import numpy as np

CACHE_SCHEMA = "exact_video_preprocessing_one_clip_v1"


class VideoPreprocessingCache:
    def __init__(self, processor):
        self.processor = processor
        self._entry = None
        self._lock = threading.Lock()
        self.last_hit = False

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def _key(self, videos, kwargs):
        if not isinstance(videos, list) or not videos:
            raise TypeError("Unsupported video container")
        fingerprints = []
        for video in videos:
            if not isinstance(video, np.ndarray) or video.dtype != np.uint8 or video.ndim != 4:
                raise TypeError("Unsupported video representation")
            fingerprints.append(
                (video.shape, video.dtype.str, hashlib.sha256(video.tobytes()).hexdigest())
            )
        return json.dumps(
            [fingerprints, kwargs, self.processor.to_dict()], sort_keys=True, allow_nan=False
        )

    def __call__(self, videos, **kwargs):
        with self._lock:
            self.last_hit = False
            try:
                key = self._key(videos, kwargs)
            except (TypeError, ValueError):
                self._entry = None
                return self.processor(videos, **kwargs)
            if self._entry is not None and key == self._entry[0]:
                self.last_hit = True
                return copy.deepcopy(self._entry[1])
            self._entry = None
            result = self.processor(videos, **kwargs)
            self._entry = (key, copy.deepcopy(result))
            return result
