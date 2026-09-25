# DenseCore Third-Party Notices

This inventory covers third-party components distributed by DenseCore Python
source distributions and wheels. Build-only tools, caches, and server-only
dependencies are not part of this inventory.

## C and C++ components

### ggml

- Bundled source path: `densecore/core_src/third_party/ggml`
- Distributed in Python source distributions and native wheels
- License: MIT
- Included license text: `licenses/ggml_LICENSE.txt`

### SIMDe, Highway, mimalloc, and spdlog

DenseCore also compiles the following C++ dependencies into `libdensecore` as
header-only or static components:

- SIMDe 0.8.2 (MIT)
- Highway 1.2.0 (Apache 2.0 and BSD-3-Clause components)
- mimalloc 2.1.2 (MIT)
- spdlog 1.12.0 (MIT; its license also records the bundled fmt dependency)

Their upstream license files are packaged under `licenses/`.

## Distribution contents

DenseCore wheels and source distributions carry:

- the Apache 2.0 project `LICENSE`
- this repository `NOTICE`
- this `THIRD_PARTY_NOTICES.md`
- packaged copies of the ggml, SIMDe, Highway, mimalloc, and spdlog licenses
