#include "common/backend.h"

namespace arbcycle {
namespace {
[[maybe_unused]] constexpr const char* kBackendName = "x86_sse";
}

// TODO(x86_sse):
//   - add the real kernel entry points for this ISA / platform variant
//   - add per-kernel dispatch glue as needed
//   - keep the exported Python surface stable; swap implementations underneath

} // namespace arbcycle
