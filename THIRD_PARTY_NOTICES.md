# DenseCore Third-Party Notices

This inventory is intentionally limited to components that are actually
distributed by DenseCore release surfaces:

- repository source checkouts and release source bundles that include submodules
- Python source distributions and wheels
- release tarballs
- container images built from this repository

Build-only local tools, editor caches, and machine-specific artifacts are not
part of this inventory.

## C and C++ components

### ggml

- Path: `core/third_party/ggml`
- Distributed in:
  - recursive source checkouts and release source bundles that include submodules
  - Python source distributions
  - release archives and container images via `libggml*`
- License: MIT
- Included license text:
  - `core/third_party/ggml/LICENSE`
  - packaged copies under `third_party/ggml/`

### SIMDe, Highway, mimalloc, and spdlog

DenseCore also compiles the following C++ dependencies into `libdensecore` as
header-only or static components:

- SIMDe 0.8.2 (MIT)
- Highway 1.2.0 (Apache 2.0 and BSD-3-Clause components)
- mimalloc 2.1.2 (MIT)
- spdlog 1.12.0 (MIT; its license also records the bundled fmt dependency)

Release archives and container images carry their upstream license files under
`third_party/cpp/`. Python distributions carry the same texts under
`licenses/`.

## Helm chart component

The DenseCore Helm chart depends on the public DenseCloud `dense-base` OCI chart
(`oci://ghcr.io/denseai/charts/dense-base:1.1.0`), which is licensed under
Apache 2.0. DenseCore source checkouts keep the corresponding license and NOTICE
texts under `charts/densecore/licenses/`, and packaged chart artifacts vendor
the resolved dependency at package time from that public OCI source.

## Go server components

The DenseCore server binary is built from `server/cmd/densecore` and statically
links its compiled Go dependency closure. Release archives and container images
ship license and notice texts for those modules under `third_party/go/`,
collected from the resolved module cache at build time.

The generated bundle is the authoritative inventory; it is derived from the
command dependency closure instead of a manually maintained module list.

## Python package notes

DenseCore wheels and source distributions carry:

- the Apache 2.0 project `LICENSE`
- this repository `NOTICE`
- this `THIRD_PARTY_NOTICES.md`
- packaged copies of the ggml, SIMDe, Highway, mimalloc, and spdlog licenses

## Build-time collection rule

Release bundles collect third-party Go module license and notice files from the
resolved module graph returned by:

```bash
cd server
go list -mod=mod -deps ./cmd/densecore
```

The collection helper only copies license/notice files for modules used by that
command and fails if a used third-party module has no discoverable license text.
