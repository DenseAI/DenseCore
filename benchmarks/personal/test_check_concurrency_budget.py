"""Pure gates only; these tests are not real c8/runtime qualification."""
import time
import threading
import pytest
from check_concurrency_budget import (DetailedMetrics, parity_errors, plan_errors,
                                      pressure_errors, rejection_valid, timed_daemon)


def negative(**updates):
    return dict(label='negative', request_id=-3, callback_events=0, completion_events=0,
                submit_error='native request admission required_bytes=134217729 available_bytes=134217728',
                **updates)


def test_explicit_runtime_cap_context_threads_and_none_required():
    plan = dict(effective_context=512, effective_concurrency=8, effective_threads=8, spec_selected=1)
    assert not plan_errors(plan, 512)
    for key in plan:
        changed = dict(plan, **{key: 0})
        assert len(plan_errors(changed, 512)) == 1


def test_sampled_ids_not_text_or_native_count_alone():
    row = dict(label='c8', terminal_events=1, completion_events=1, generated_tokens=128,
               finish_reason=2, token_ids=list(range(128)))
    assert not parity_errors([row], list(range(128)))
    changed = dict(row, token_ids=list(range(127))+[999])
    assert 'sampled IDs' in parity_errors([changed], row['token_ids'])[0]


@pytest.mark.parametrize('field,value', [('terminal_events',2),('completion_events',0),
                                         ('generated_tokens',127),('finish_reason',1)])
def test_early_eos_or_duplicate_terminal_cannot_pass_length_parity(field, value):
    row = dict(label='c8', terminal_events=1, completion_events=1, generated_tokens=128,
               finish_reason=2, token_ids=list(range(128)))
    row[field] = value
    assert parity_errors([row], row['token_ids'])


def test_stop_can_fit_but_prompt_rejection_proves_bounded_admission():
    accepted = dict(label='stop', request_id=11, callback_events=1, completion_events=1)
    assert not pressure_errors([accepted, negative()])
    assert 'not established' in pressure_errors([accepted, accepted])[0]


@pytest.mark.parametrize('field,value', [('request_id',1),('callback_events',1),
    ('completion_events',1),('submit_error','required_bytes=10 available_bytes=10'),
    ('submit_error','out of memory')])
def test_negative_requires_synchronous_sized_error_and_no_callbacks(field, value):
    row = negative()
    row[field] = value
    assert not rejection_valid(row)


def test_late_callback_invalidates_previously_valid_rejection():
    row = negative()
    assert rejection_valid(row)
    row['callback_events'] += 1
    assert pressure_errors([row])


def test_daemon_timeout_does_not_wait_for_native_return():
    release = threading.Event()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            timed_daemon(lambda: release.wait(10), .02)
        assert time.monotonic()-started < 1
    finally:
        release.set()


def test_daemon_error_preserves_original_failure():
    def fail(): raise ValueError('producer failure')
    with pytest.raises(ValueError, match='producer failure'):
        timed_daemon(fail, 1)


def test_native_metrics_uses_platform_long_and_preserves_field_order():
    # Full header ABI order; actual runtime will provide the measurements.
    fields = [key for key, _ in DetailedMetrics._fields_]
    assert fields[:5] == ['active_requests','total_requests','completed_requests','failed_requests','pending_requests']
    assert fields[-5:] == ['kv_cache_usage_percent','avg_batch_size','current_batch_size','oom_errors','timeout_errors']


def test_truncated_callback_baseline_cannot_validate_itself():
    row = dict(label='baseline', terminal_events=1, completion_events=1, generated_tokens=128,
               finish_reason=2, token_ids=list(range(127)))
    assert 'count' in parity_errors([row], row['token_ids'])[0]
