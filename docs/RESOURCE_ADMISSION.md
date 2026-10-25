# Request admission memory

The native runtime partitions half of the existing initialization memory plan's
allocator reserve for request admission. It does not add a second memory budget
or shrink the effective context. Plans from models in the process contribute to
one shared byte counter. Standalone graph adapters without a model plan derive
admission capacity from detected finite remaining memory divided by 32, capped
at 128 MiB; a detected unlimited budget uses that cap, while zero or unknown
memory permits no admission. This fallback does not represent a committed graph
scratch reserve. Graph scratch allocation follows its separate graph policy.
Allocation refusal returns the existing C status
`DENSECORE_STATUS_OUT_OF_MEMORY` immediately, before publication; rejected
requests receive no callback. A batch rejected at its first child stops there
without callbacks. Once its first child is accepted, the positive-return batch
produces one terminal callback per child, including explicit error callbacks for
later admission failures. Accepted requests retain the normal terminal
callback contract. Go removes failed submission registrations; Python raises an
exception without waiting for a callback that will never arrive.

Admission reserves owned prompt/token/constraint storage before copying or
tokenizing caller input. Arithmetic is checked for overflow. Caller-owned text
is scanned only up to the remaining byte budget, and hostile token counts can
be rejected without reading the caller token array. Text reservation includes
conservative tokenizer and template overhead. Generic graph inputs are sized
and validated before tensor copies. Rendered-chat snapshot preparation holds
an additional temporary reservation while both snapshot and request exist.

The reservation also includes conservative context-bounded token histories and
output buffering. Streaming does not guarantee immediate consumption: native
result events retain the reservation through callback delivery, including
queued events that outlive their retired request. Output reservation uses the
model's largest vocabulary piece and the effective context ceiling. This can
reject a request earlier than its eventual measured peak would require.

Each pooled Request object keeps a metadata charge for its entire lifetime,
which bounds request count through the same byte budget. Retiring a request
releases its payload container capacities and payload lease while retaining
its small reusable metadata charge. Cancellation and shutdown use this same
retirement path. Queued output can keep a payload charge until callbacks drain.
The policy meters native ownership; it does not meter Go or Python allocations
made after callbacks copy outputs, or the caller's original request buffers.

`DENSECORE_MAX_NUM_SEQS` separately bounds requests with execution KV/recurrent
state. The Go `DENSECORE_SERVER_INFLIGHT` setting limits only requests waiting in
the dispatch heap. Asynchronous native submission can drain that heap before
generation completes, so it is not an active-request count. A full Go heap
returns HTTP 503 with `service_unavailable` and `Retry-After: 1`, before SSE starts.
Neither a configured sequence cap of eight nor passing admission tests proves
that a physical batch of eight executed or that throughput improved.
