# Fork Conflict Notes

## Fork Conflict Map

- `scheduler.py` is fork-customized. Grep `# fork:` before and after merges.
- Other known fork-touched areas include `engine_pool.py`, `engine_core.py`,
  `engine/dflash.py`, `engine/vlm.py`, `process_memory_enforcer.py`,
  `cache/prefix_cache.py`, `cache/paged_cache.py`, `cache/paged_ssd_cache.py`,
  `server.py`, `settings.py`, `utils/image.py`, `engine/tts.py`,
  `engine/sts.py`, and `admin/oq_manager.py`.
- For dead-code-removal conflicts, keep the local deletion unless upstream has
  added a real caller. Check with `git grep -nw <symbol> origin/main` before
  restoring deleted code.
- If upstream reintroduces a previously deleted symbol with a new caller or
  regression test, restore upstream instead of preserving the fork deletion.
  Example from the 0.4.4rc2 bump: `load_text_model` came back with
  `trust_remote_code` handling and an upstream test, so it was restored.
- If a fork-only feature is dropped, prune orphaned fork tests only after
  confirming upstream does not contain that test or behavior.
- Fork tests that bind `Scheduler` methods onto `SimpleNamespace` may need new
  upstream helper methods added to the namespace. Example: after 0.4.4rc2,
  `_step_prefill_chunk` tests needed `_periodic_clear_threshold_bytes`.

## Recurring merge recipes

- For streaming or keepalive errors, preserve upstream special cases for
  `PrefillMemoryExceededError`, then use the fork sanitizer for generic
  exceptions.
- Keep fork async token-counting and grammar compile helpers over upstream
  synchronous request-path calls when those conflict.
- For `Scheduler._async_store_cache_worker`, combine upstream's pressure-mode
  `hot_cache_write_back=False` path with the fork's per-block SSD extraction
  lock. Do not re-expand the lock to cover the whole multi-block `store_cache`
  call unless upstream has moved the real buffer access protection deeper.
- For `/v1/completions` prompt validation, keep
  `await _encode_tokens_for_engine(...)` so tokenization stays off the request
  path, but also preserve upstream's `prompt_token_ids_by_prompt` reuse for
  streaming thinking-prefix handling.
- For VLM MTP decode, keep upstream adapter proxy / `prompt_tokens` plumbing and
  preserve the fork-injectable `clear_cache` callback used by tests and
  per-round reclaim.
- For `engine_core.py` conflicts around MLX executor thread init and teardown
  reclaim, keep upstream `_final_engine_thread_reclaim` (`gc.collect()` plus
  `_sync_and_clear_cache(stream)`) and keep the fork helper that patches
  generation streams in `mlx_lm.generate`, `mlx_vlm.generate`, and
  `omlx.scheduler`. Do not regress to the upstream-only `mlx_lm`/scheduler
  patch.
- For `models/base_model.py` pooling-helper conflicts, keep both the fork
  `max_pooling` / `mean_sqrt_len_pooling` helpers used by `xlm_roberta` and
  upstream `last_token_pool` used by Qwen2/Qwen3 embedding models.
- For Responses API reasoning-token usage conflicts, keep fork async token
  counting (`await _count_prompt_tokens_for_engine(engine, reasoning_text)`)
  instead of direct `len(engine.tokenizer.encode(...))`, so tokenization stays
  off the request path.
