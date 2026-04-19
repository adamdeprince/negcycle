#include "common/backend.h"

namespace arbcycle {
namespace {
[[maybe_unused]] constexpr const char* kBackendName = "linux_powerpc64_vsx";
}

// TODO(linux_powerpc64_vsx):
//   - add the real kernel entry points for this ISA / platform variant
//   - add per-kernel dispatch glue as needed
//   - keep the exported Python surface stable; swap implementations underneath

} // namespace arbcycle
