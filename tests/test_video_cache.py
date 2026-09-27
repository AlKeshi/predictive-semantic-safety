import copy

import numpy as np
import pytest

from pss.predictors.video_cache import VideoPreprocessingCache


class Processor:
    def __init__(self):
        self.calls = 0
        self.size = 32
        self.fail = False

    def to_dict(self):
        return {"size": self.size}

    def __call__(self, videos, **kwargs):
        self.calls += 1
        if self.fail:
            raise ValueError("Invalid video")
        return {"pixels": videos[0].copy(), "metadata": copy.deepcopy(kwargs)}


def inputs():
    return (
        [np.zeros((4, 8, 8, 3), dtype=np.uint8)],
        {
            "video_metadata": [{"frames_indices": [0, 1, 2, 3], "fps": 2.0}],
            "size": {"longest_edge": 1024},
            "do_sample_frames": False,
        },
    )


def test_identical_content_reuses_processing_without_sharing_outputs():
    processor = Processor()
    cache = VideoPreprocessingCache(processor)
    videos, options = inputs()
    first = cache(videos, **options)
    first["pixels"][:] = 255
    first["metadata"]["video_metadata"][0]["fps"] = -1
    second = cache(copy.deepcopy(videos), **copy.deepcopy(options))
    assert cache.last_hit and processor.calls == 1
    assert not second["pixels"].any()
    assert second["metadata"]["video_metadata"][0]["fps"] == 2.0
    second["pixels"][:] = 100
    assert not cache(videos, **options)["pixels"].any()


@pytest.mark.parametrize("change", ["pixel", "time", "fps", "size", "sample", "shape", "defaults"])
def test_changed_inputs_or_settings_invalidate(change):
    processor = Processor()
    cache = VideoPreprocessingCache(processor)
    videos, options = inputs()
    cache(videos, **options)
    if change == "pixel":
        videos[0][0, 0, 0, 0] = 1
    elif change == "time":
        options["video_metadata"][0]["frames_indices"][-1] = 4
    elif change == "fps":
        options["video_metadata"][0]["fps"] = 3.0
    elif change == "size":
        options["size"]["longest_edge"] = 2048
    elif change == "sample":
        options["do_sample_frames"] = True
    elif change == "shape":
        videos[0] = videos[0].reshape(4, 4, 16, 3)
    else:
        processor.size = 64
    cache(videos, **options)
    assert not cache.last_hit and processor.calls == 2


def test_bounded_to_one_clip_and_failed_processing_is_not_cached():
    processor = Processor()
    cache = VideoPreprocessingCache(processor)
    videos, options = inputs()
    cache(videos, **options)
    changed = [videos[0] + 1]
    cache(changed, **options)
    cache(videos, **options)
    assert processor.calls == 3
    processor.fail = True
    with pytest.raises(ValueError, match="Invalid video"):
        cache(changed, **options)
    processor.fail = False
    cache(changed, **options)
    assert not cache.last_hit and processor.calls == 5


def test_unsupported_options_bypass_cache_instead_of_guessing_identity():
    processor = Processor()
    cache = VideoPreprocessingCache(processor)
    videos, options = inputs()
    options["unknown_option"] = object()
    cache(videos, **options)
    cache(videos, **options)
    assert not cache.last_hit and processor.calls == 2
