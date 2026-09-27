import os
from types import SimpleNamespace

import pytest

from pss.vla.omnivla_loading import OmniVLALoading


@pytest.mark.skipif(not os.environ.get("OMNIVLA_SOURCE"), reason="Requires official OmniVLA source")
def test_official_loader_imports_before_cuda_check(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loader = OmniVLALoading()
    loader.config = SimpleNamespace(source_dir=os.environ["OMNIVLA_SOURCE"], device="cuda")
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        loader._load_model()
