"""Unit coverage for oMLX's mlx-lm GenerationBatch integration patch."""

import mlx.core as mx

from omlx import scheduler
from omlx.request import SamplingParams


def _reset_sampler_batch_stats():
    for key, value in scheduler._SAMPLER_BATCH_STATS.items():
        if isinstance(value, list):
            scheduler._SAMPLER_BATCH_STATS[key] = []
        else:
            scheduler._SAMPLER_BATCH_STATS[key] = 0


def test_sampler_key_canonicalizes_argmax_params():
    a = scheduler._sampler_key(
        SamplingParams(temperature=0.0, top_p=0.1, top_k=0),
        [1, 2, 3],
    )
    b = scheduler._sampler_key(
        SamplingParams(temperature=0.0, top_p=0.99, top_k=128),
        [9],
    )
    c = scheduler._sampler_key(
        SamplingParams(temperature=0.1, top_p=0.99, top_k=128),
        [9],
    )

    assert a == b
    assert a != c


def test_grouped_sample_batches_tagged_equivalent_samplers():
    _reset_sampler_batch_stats()
    calls = []

    def make_sampler(name):
        def sampler(logprobs):
            calls.append((name, tuple(logprobs.shape)))
            return mx.argmax(logprobs, axis=-1)

        return scheduler._tag_sampler(sampler, ("sampler", name[0]))

    sampler_a1 = make_sampler("a1")
    sampler_a2 = make_sampler("a2")
    sampler_b = make_sampler("b")
    sampler_a3 = make_sampler("a3")

    class Batch:
        uids = [10, 11, 12, 13]
        samplers = [sampler_a1, sampler_a2, sampler_b, sampler_a3]
        fallback_sampler = sampler_b

    logprobs = mx.array(
        [
            [0.0, 1.0, 4.0],
            [3.0, 1.0, 0.0],
            [0.0, 5.0, 1.0],
            [2.0, 3.0, 1.0],
        ]
    )

    sampled = scheduler._grouped_sample(Batch(), logprobs)

    assert sampled.tolist() == [2, 0, 1, 1]
    assert calls == [("a1", (3, 3)), ("b", (1, 3))]
    assert scheduler._SAMPLER_BATCH_STATS["last_batch_size"] == 4
    assert scheduler._SAMPLER_BATCH_STATS["last_group_count"] == 2
    assert scheduler._SAMPLER_BATCH_STATS["last_group_sizes"] == [3, 1]
    assert scheduler._SAMPLER_BATCH_STATS["avoided_sampler_calls"] == 2


def test_grouped_sample_keeps_distinct_untagged_samplers_separate():
    _reset_sampler_batch_stats()
    calls = []

    def sampler_a(logprobs):
        calls.append(("a", tuple(logprobs.shape)))
        return mx.argmax(logprobs, axis=-1)

    def sampler_b(logprobs):
        calls.append(("b", tuple(logprobs.shape)))
        return mx.argmax(logprobs, axis=-1)

    class Batch:
        uids = [10, 11]
        samplers = [sampler_a, sampler_b]
        fallback_sampler = sampler_a

    sampled = scheduler._grouped_sample(
        Batch(),
        mx.array([[0.0, 2.0], [3.0, 1.0]]),
    )

    assert sampled.tolist() == [1, 0]
    assert calls == [("a", (1, 2)), ("b", (1, 2))]
    assert scheduler._SAMPLER_BATCH_STATS["last_group_count"] == 2
    assert scheduler._SAMPLER_BATCH_STATS["avoided_sampler_calls"] == 0


def test_generation_batch_patch_reports_install_mode():
    patch = scheduler._SAMPLER_BATCH_PATCH

    assert patch["installed"] is True
    assert patch["mode"] in {"grouped", "compat"}
    if patch["compatible"]:
        assert patch["mode"] == "grouped"
        assert patch["disabled_reason"] is None
    else:
        assert patch["mode"] == "compat"
        assert patch["disabled_reason"]
