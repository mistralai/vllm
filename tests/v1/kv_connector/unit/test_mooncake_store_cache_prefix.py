# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from typing import Any

import pytest
from transformers import OPTConfig

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    KVTransferConfig,
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.worker import (
    _resolve_cache_prefix,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def model_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    OPTConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        ffn_dim=128,
        word_embed_proj_dim=64,
    ).save_pretrained(tmp_path)
    monkeypatch.setenv("VLLM_IMAGE_TAG", "vllm:test-build-1")
    monkeypatch.delenv("VLLM_BUILD_COMMIT", raising=False)
    return tmp_path


def make_config(
    model_path: Path, cache_prefix: str = "auto", dtype: str = "float16", **kwargs: Any
) -> VllmConfig:
    return VllmConfig(
        model_config=ModelConfig(model=str(model_path), dtype=dtype),
        device_config=DeviceConfig("cpu"),
        kv_transfer_config=KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"cache_prefix": cache_prefix},
        ),
        **kwargs,
    )


@pytest.mark.parametrize("prefix", ["", "custom-fp4", "AUTO"])
def test_literal_prefix_is_unchanged(model_path: Path, prefix: str) -> None:
    assert _resolve_cache_prefix(make_config(model_path, prefix)) == prefix


def test_auto_prefix_is_stable_across_replicas(model_path: Path) -> None:
    prefix = _resolve_cache_prefix(make_config(model_path))
    assert prefix.startswith("auto-")
    # Scheduler capacity is not part of the namespace.
    other = make_config(
        model_path,
        scheduler_config=SchedulerConfig(
            max_model_len=2048,
            is_encoder_decoder=False,
            max_num_batched_tokens=4096,
            max_num_seqs=8,
        ),
    )
    assert _resolve_cache_prefix(other) == prefix


@pytest.mark.parametrize(
    "variable,value",
    [("VLLM_IMAGE_TAG", "vllm:test-build-2"), ("VLLM_BUILD_COMMIT", "abc123")],
)
def test_build_identity_change_isolates(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    config = make_config(model_path)
    old = _resolve_cache_prefix(config)
    if variable == "VLLM_BUILD_COMMIT":
        monkeypatch.delenv("VLLM_IMAGE_TAG")
    monkeypatch.setenv(variable, value)
    assert _resolve_cache_prefix(config) != old


def test_model_or_kv_format_change_isolates(model_path: Path) -> None:
    base = _resolve_cache_prefix(make_config(model_path))
    assert _resolve_cache_prefix(make_config(model_path, dtype="bfloat16")) != base
    fp8_kv = make_config(model_path, cache_config=CacheConfig(cache_dtype="fp8"))
    assert _resolve_cache_prefix(fp8_kv) != base


@pytest.mark.parametrize(
    "attention_kwargs",
    [
        {"backend": AttentionBackendEnum.FLASH_ATTN},
        {"flash_attn_version": 3},
    ],
)
def test_attention_change_isolates(
    model_path: Path, attention_kwargs: dict[str, Any]
) -> None:
    base = _resolve_cache_prefix(make_config(model_path))
    config = make_config(
        model_path, attention_config=AttentionConfig(**attention_kwargs)
    )
    assert _resolve_cache_prefix(config) != base


@pytest.mark.parametrize(
    "draft_kwargs",
    [
        {"revision": "different-revision"},
        {"code_revision": "different-code-revision"},
        {"attention_backend": AttentionBackendEnum.FLASH_ATTN},
        {"num_speculative_tokens": 2},
        {"disable_lookahead_block_hashes": True},
    ],
)
def test_draft_configuration_change_isolates(
    model_path: Path, draft_kwargs: dict[str, Any]
) -> None:
    def config(**overrides: Any) -> VllmConfig:
        target = make_config(model_path)
        return make_config(
            model_path,
            speculative_config=SpeculativeConfig(
                **{
                    "method": "draft_model",
                    "model": str(model_path),
                    "num_speculative_tokens": 1,
                    "target_model_config": target.model_config,
                    "target_parallel_config": target.parallel_config,
                    **overrides,
                }
            ),
        )

    base = _resolve_cache_prefix(config())
    assert _resolve_cache_prefix(config()) == base
    assert _resolve_cache_prefix(config(**draft_kwargs)) != base


@pytest.mark.parametrize("image_tag", ["", "local/vllm-openai:dev"])
def test_auto_requires_build_identity(
    model_path: Path, monkeypatch: pytest.MonkeyPatch, image_tag: str
) -> None:
    monkeypatch.setenv("VLLM_IMAGE_TAG", image_tag)
    with pytest.raises(ValueError, match="VLLM_IMAGE_TAG or VLLM_BUILD_COMMIT"):
        _resolve_cache_prefix(make_config(model_path))
