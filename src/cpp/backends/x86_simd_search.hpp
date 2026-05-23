#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#if defined(__loongarch_sx)
#include <lsxintrin.h>
#endif
#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif

#include "common/detector_base.h"

namespace negcycle {

#if defined(__SSE__)
struct SseSimdTraits {
  using Vec = __m128;
  static constexpr int lanes = 4;
  static constexpr std::size_t alignment = 16;

  [[nodiscard]] static Vec set1(float value) noexcept { return _mm_set1_ps(value); }
  [[nodiscard]] static Vec load(const float* ptr) noexcept { return _mm_loadu_ps(ptr); }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept { return _mm_add_ps(lhs, rhs); }
  [[nodiscard]] static Vec lane_indices(int first) noexcept {
    return _mm_add_ps(_mm_set1_ps(static_cast<float>(first)),
                      _mm_setr_ps(0.0f, 1.0f, 2.0f, 3.0f));
  }
  [[nodiscard]] static int neq_mask(Vec lhs, Vec rhs) noexcept {
    return _mm_movemask_ps(_mm_cmpneq_ps(lhs, rhs));
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    return _mm_movemask_ps(_mm_cmplt_ps(value, _mm_setzero_ps()));
  }
  static void store(float* ptr, Vec value) noexcept { _mm_storeu_ps(ptr, value); }
};
#endif

#if defined(__AVX__)
struct AvxSimdTraits {
  using Vec = __m256;
  static constexpr int lanes = 8;
  static constexpr std::size_t alignment = 32;

  [[nodiscard]] static Vec set1(float value) noexcept { return _mm256_set1_ps(value); }
  [[nodiscard]] static Vec load(const float* ptr) noexcept { return _mm256_loadu_ps(ptr); }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept { return _mm256_add_ps(lhs, rhs); }
  [[nodiscard]] static Vec lane_indices(int first) noexcept {
    return _mm256_add_ps(_mm256_set1_ps(static_cast<float>(first)),
                         _mm256_setr_ps(0.0f, 1.0f, 2.0f, 3.0f,
                                        4.0f, 5.0f, 6.0f, 7.0f));
  }
  [[nodiscard]] static int neq_mask(Vec lhs, Vec rhs) noexcept {
    return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OQ));
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    return _mm256_movemask_ps(_mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_LT_OQ));
  }
  static void store(float* ptr, Vec value) noexcept { _mm256_storeu_ps(ptr, value); }
};
#endif

#if defined(__AVX2__)
struct Avx2SimdTraits {
  using Vec = __m256;
  static constexpr int lanes = 8;
  static constexpr std::size_t alignment = 32;

  [[nodiscard]] static Vec set1(float value) noexcept { return _mm256_set1_ps(value); }
  [[nodiscard]] static Vec load(const float* ptr) noexcept { return _mm256_loadu_ps(ptr); }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept { return _mm256_add_ps(lhs, rhs); }
  [[nodiscard]] static Vec lane_indices(int first) noexcept {
    const __m256i offsets = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(first), offsets));
  }
  [[nodiscard]] static int neq_mask(Vec lhs, Vec rhs) noexcept {
    return _mm256_movemask_ps(_mm256_cmp_ps(lhs, rhs, _CMP_NEQ_OQ));
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    return _mm256_movemask_ps(_mm256_cmp_ps(value, _mm256_setzero_ps(), _CMP_LT_OQ));
  }
  static void store(float* ptr, Vec value) noexcept { _mm256_storeu_ps(ptr, value); }
};
#endif

#if defined(__AVX512F__)
struct Avx512SimdTraits {
  using Vec = __m512;
  static constexpr int lanes = 16;
  static constexpr std::size_t alignment = 64;

  [[nodiscard]] static Vec set1(float value) noexcept { return _mm512_set1_ps(value); }
  [[nodiscard]] static Vec load(const float* ptr) noexcept { return _mm512_loadu_ps(ptr); }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept { return _mm512_add_ps(lhs, rhs); }
  [[nodiscard]] static Vec lane_indices(int first) noexcept {
    const __m512i offsets =
        _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                          8, 9, 10, 11, 12, 13, 14, 15);
    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_set1_epi32(first), offsets));
  }
  [[nodiscard]] static int neq_mask(Vec lhs, Vec rhs) noexcept {
    return static_cast<int>(_mm512_cmp_ps_mask(lhs, rhs, _CMP_NEQ_OQ));
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    return static_cast<int>(_mm512_cmp_ps_mask(value, _mm512_setzero_ps(), _CMP_LT_OQ));
  }
  static void store(float* ptr, Vec value) noexcept { _mm512_storeu_ps(ptr, value); }
};
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
struct NeonSimdTraits {
  using Vec = float32x4_t;
  static constexpr int lanes = 4;
  static constexpr std::size_t alignment = 16;

  [[nodiscard]] static Vec set1(float value) noexcept {
    return vdupq_n_f32(value);
  }
  [[nodiscard]] static Vec load(const float* ptr) noexcept {
    return vld1q_f32(ptr);
  }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept {
    return vaddq_f32(lhs, rhs);
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    const uint32x4_t cmp = vcltq_f32(value, vdupq_n_f32(0.0f));
    const uint32x4_t weights = {1u, 2u, 4u, 8u};
    const uint32x4_t weighted = vandq_u32(cmp, weights);
#if defined(__aarch64__) || defined(_M_ARM64)
    return static_cast<int>(vaddvq_u32(weighted));
#else
    const uint32x2_t halves =
        vadd_u32(vget_low_u32(weighted), vget_high_u32(weighted));
    return static_cast<int>(vget_lane_u32(vpadd_u32(halves, halves), 0));
#endif
  }
  static void store(float* ptr, Vec value) noexcept {
    vst1q_f32(ptr, value);
  }
};
#endif

#if defined(__ARM_FEATURE_SVE)
struct SveSimdTraits {};
#endif

#if defined(__loongarch_sx)
struct LsxSimdTraits {
  using Vec = __m128;
  static constexpr int lanes = 4;
  static constexpr std::size_t alignment = 16;

  [[nodiscard]] static Vec set1(float value) noexcept {
    return (__m128)__lsx_vldrepl_w(&value, 0);
  }
  [[nodiscard]] static Vec load(const float* ptr) noexcept {
    return (__m128)__lsx_vld(const_cast<float*>(ptr), 0);
  }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept {
    return __lsx_vfadd_s(lhs, rhs);
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    const __m128 zero = (__m128)__lsx_vldi(0);
    const __m128i cmp = __lsx_vfcmp_clt_s(value, zero);
    return __lsx_vpickve2gr_w(__lsx_vmskltz_w(cmp), 0) & 0x0f;
  }
  static void store(float* ptr, Vec value) noexcept {
    __lsx_vst((__m128i)value, ptr, 0);
  }
};
#endif

#if defined(__loongarch_asx)
struct LasxSimdTraits {
  using Vec = __m256;
  static constexpr int lanes = 8;
  static constexpr std::size_t alignment = 32;

  [[nodiscard]] static Vec set1(float value) noexcept {
    return (__m256)__lasx_xvldrepl_w(&value, 0);
  }
  [[nodiscard]] static Vec load(const float* ptr) noexcept {
    return (__m256)__lasx_xvld(const_cast<float*>(ptr), 0);
  }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept {
    return __lasx_xvfadd_s(lhs, rhs);
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    const __m256 zero = (__m256)__lasx_xvldi(0);
    const __m256i cmp = __lasx_xvfcmp_clt_s(value, zero);
    const __m256i packed = __lasx_xvmskltz_w(cmp);
    const int low = __lasx_xvpickve2gr_w(packed, 0) & 0x0f;
    const int high = __lasx_xvpickve2gr_w(packed, 4) & 0x0f;
    return low | (high << 4);
  }
  static void store(float* ptr, Vec value) noexcept {
    __lasx_xvst((__m256i)value, ptr, 0);
  }
};
#endif

struct PortableSimd128Traits {
  typedef float Vec __attribute__((vector_size(16)));
  static constexpr int lanes = 4;
  static constexpr std::size_t alignment = 16;

  [[nodiscard]] static Vec set1(float value) noexcept {
    Vec out{};
    for (int i = 0; i < lanes; ++i) {
      out[i] = value;
    }
    return out;
  }
  [[nodiscard]] static Vec load(const float* ptr) noexcept {
    Vec out{};
    __builtin_memcpy(&out, ptr, sizeof(out));
    return out;
  }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept { return lhs + rhs; }
  [[nodiscard]] static Vec lane_indices(int first) noexcept {
    Vec out{};
    for (int i = 0; i < lanes; ++i) {
      out[i] = static_cast<float>(first + i);
    }
    return out;
  }
  [[nodiscard]] static int neq_mask(Vec lhs, Vec rhs) noexcept {
    int mask = 0;
    for (int i = 0; i < lanes; ++i) {
      if (lhs[i] != rhs[i]) {
        mask |= 1 << i;
      }
    }
    return mask;
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    int mask = 0;
    for (int i = 0; i < lanes; ++i) {
      if (value[i] < 0.0f) {
        mask |= 1 << i;
      }
    }
    return mask;
  }
  static void store(float* ptr, Vec value) noexcept {
    __builtin_memcpy(ptr, &value, sizeof(value));
  }
};

#if !defined(__x86_64__) && !defined(__i386__) || defined(__AVX__)
struct PortableSimd256Traits {
  typedef float Vec __attribute__((vector_size(32)));
  static constexpr int lanes = 8;
  static constexpr std::size_t alignment = 32;

  [[nodiscard]] static Vec set1(float value) noexcept {
    Vec out{};
    for (int i = 0; i < lanes; ++i) {
      out[i] = value;
    }
    return out;
  }
  [[nodiscard]] static Vec load(const float* ptr) noexcept {
    Vec out{};
    __builtin_memcpy(&out, ptr, sizeof(out));
    return out;
  }
  [[nodiscard]] static Vec add(Vec lhs, Vec rhs) noexcept { return lhs + rhs; }
  [[nodiscard]] static Vec lane_indices(int first) noexcept {
    Vec out{};
    for (int i = 0; i < lanes; ++i) {
      out[i] = static_cast<float>(first + i);
    }
    return out;
  }
  [[nodiscard]] static int neq_mask(Vec lhs, Vec rhs) noexcept {
    int mask = 0;
    for (int i = 0; i < lanes; ++i) {
      if (lhs[i] != rhs[i]) {
        mask |= 1 << i;
      }
    }
    return mask;
  }
  [[nodiscard]] static int lt_zero_mask(Vec value) noexcept {
    int mask = 0;
    for (int i = 0; i < lanes; ++i) {
      if (value[i] < 0.0f) {
        mask |= 1 << i;
      }
    }
    return mask;
  }
  static void store(float* ptr, Vec value) noexcept {
    __builtin_memcpy(ptr, &value, sizeof(value));
  }
};
#endif

// Scan policy: encapsulates how `scan_last_hop` reads the dense weight rows
// and how `find_best_arbitrage` derives a pruning threshold from the running
// best candidate. The default policy (`LooseScanPolicy`) reproduces the
// historical behavior for narrow SIMD: scan up to `n`, handle the unaligned
// end with a scalar tail, no threshold tightening.
//
// AVX-512 uses `PaddedTightScanPolicy`. Its kernel relies on the dense
// weights being padded to `padded_stride` with +inf, so it can run the SIMD
// loop without a scalar tail, AND it folds the running best weight into the
// broadcast prefix so the SIMD compare prunes any candidate that can't beat
// `best`. SVE uses the same padded/tight idea with SVE predicates for the
// vector tail. Benchmarks on the narrower fixed-width SIMD backends showed
// this was a net loss there.

struct LooseScanPolicy {
  // No padding: dense rows have length `n`, the kernel handles the unaligned
  // end with a scalar tail.
  static constexpr int kPadMultiple = 1;

  template <typename DenseWeights>
  [[nodiscard]] static int simd_scan_end(const DenseWeights& dense) noexcept {
    return dense.n;
  }

  template <typename BestCandidate>
  [[nodiscard]] static float threshold(const BestCandidate&) noexcept {
    return 0.0f;
  }

  template <typename Traits, int PrefixLen, typename RecordFn>
  static void scan_last_hop(const float* close_to_start,
                            const float* row_current,
                            int first_candidate,
                            int scan_end,
                            float prefix_weight,
                            float /* threshold */,
                            const int* prefix,
                            RecordFn&& record) {
    using Vec = typename Traits::Vec;

    const Vec prefix_v = Traits::set1(prefix_weight);
    const int full_mask = (1 << Traits::lanes) - 1;
    alignas(Traits::alignment) float totals[Traits::lanes];

    int j = first_candidate;
    for (; j + Traits::lanes <= scan_end; j += Traits::lanes) {
      int valid_mask = full_mask;
      for (int i = 0; i < PrefixLen; ++i) {
        const int lane = prefix[i] - j;
        if (lane >= 0 && lane < Traits::lanes) {
          valid_mask &= ~(1 << lane);
        }
      }

      const Vec total_v =
          Traits::add(prefix_v,
                      Traits::add(Traits::load(row_current + static_cast<std::size_t>(j)),
                                  Traits::load(close_to_start + static_cast<std::size_t>(j))));
      const int mask = valid_mask & Traits::lt_zero_mask(total_v);
      if (mask == 0) {
        continue;
      }

      Traits::store(totals, total_v);
      int pending = mask;
      while (pending != 0) {
        const int lane = __builtin_ctz(static_cast<unsigned int>(pending));
        pending &= pending - 1;

        int path[6] = {0, 0, 0, 0, 0, 0};
        for (int i = 0; i < PrefixLen; ++i) {
          path[i] = prefix[i];
        }
        path[PrefixLen] = j + lane;
        path[PrefixLen + 1] = prefix[0];
        record(totals[lane], PrefixLen + 1, path);
      }
    }

    for (; j < scan_end; ++j) {
      bool valid = true;
      for (int i = 0; i < PrefixLen; ++i) {
        if (j == prefix[i]) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        continue;
      }

      const float total_weight =
          prefix_weight +
          row_current[static_cast<std::size_t>(j)] +
          close_to_start[static_cast<std::size_t>(j)];
      if (!(total_weight < 0.0f)) {
        continue;
      }

      int path[6] = {0, 0, 0, 0, 0, 0};
      for (int i = 0; i < PrefixLen; ++i) {
        path[i] = prefix[i];
      }
      path[PrefixLen] = j;
      path[PrefixLen + 1] = prefix[0];
      record(total_weight, PrefixLen + 1, path);
    }
  }
};

struct PaddedTightScanPolicy {
  // Pad rows to AVX-512's lane width (16 floats) and fill the trailing cells
  // with +inf so the SIMD inner loop can run straight to `padded_stride`.
  static constexpr int kPadMultiple = 16;

  template <typename DenseWeights>
  [[nodiscard]] static int simd_scan_end(const DenseWeights& dense) noexcept {
    return dense.padded_stride;
  }

  template <typename BestCandidate>
  [[nodiscard]] static float threshold(const BestCandidate& best) noexcept {
    if (!best.have) {
      return 0.0f;
    }
    // Keep candidates within epsilon of the current best because the
    // length / lex tiebreak inside `record_best` can still flip them.
    const float widened = best.weight + ArbitrageDetectorBase::kCompareEpsilon;
    return widened < 0.0f ? widened : 0.0f;
  }

  template <typename Traits, int PrefixLen, typename RecordFn>
  static void scan_last_hop(const float* close_to_start,
                            const float* row_current,
                            int first_candidate,
                            int scan_end,
                            float prefix_weight,
                            float threshold,
                            const int* prefix,
                            RecordFn&& record) {
    using Vec = typename Traits::Vec;

    // (prefix - threshold) + row + close < 0  iff  prefix + row + close < threshold.
    // Folding threshold into the broadcast keeps the inner compare a single
    // `lt_zero_mask`. The dense weight rows are padded with +inf, so no
    // scalar tail is needed.
    const Vec prefix_v = Traits::set1(prefix_weight - threshold);
    const int full_mask = (1 << Traits::lanes) - 1;
    alignas(Traits::alignment) float totals[Traits::lanes];

    for (int j = first_candidate; j + Traits::lanes <= scan_end; j += Traits::lanes) {
      int valid_mask = full_mask;
      for (int i = 0; i < PrefixLen; ++i) {
        const int lane = prefix[i] - j;
        if (lane >= 0 && lane < Traits::lanes) {
          valid_mask &= ~(1 << lane);
        }
      }

      const Vec total_v =
          Traits::add(prefix_v,
                      Traits::add(Traits::load(row_current + static_cast<std::size_t>(j)),
                                  Traits::load(close_to_start + static_cast<std::size_t>(j))));
      const int mask = valid_mask & Traits::lt_zero_mask(total_v);
      if (mask == 0) {
        continue;
      }

      Traits::store(totals, total_v);
      int pending = mask;
      while (pending != 0) {
        const int lane = __builtin_ctz(static_cast<unsigned int>(pending));
        pending &= pending - 1;

        int path[6] = {0, 0, 0, 0, 0, 0};
        for (int i = 0; i < PrefixLen; ++i) {
          path[i] = prefix[i];
        }
        path[PrefixLen] = j + lane;
        path[PrefixLen + 1] = prefix[0];
        record(totals[lane] + threshold, PrefixLen + 1, path);
      }
    }
  }
};


#if defined(__ARM_FEATURE_SVE)
struct SvePaddedTightScanPolicy {
  // Keep the same row-padding multiple as AVX-512 for 64-byte row alignment.
  // SVE predicates guard any vector tail when the runtime vector length does
  // not divide this stride exactly.
  static constexpr int kPadMultiple = 16;
  static constexpr int kMaxFloatLanes = 64;

  template <typename DenseWeights>
  [[nodiscard]] static int simd_scan_end(const DenseWeights& dense) noexcept {
    return dense.padded_stride;
  }

  template <typename BestCandidate>
  [[nodiscard]] static float threshold(const BestCandidate& best) noexcept {
    if (!best.have) {
      return 0.0f;
    }
    const float widened = best.weight + ArbitrageDetectorBase::kCompareEpsilon;
    return widened < 0.0f ? widened : 0.0f;
  }

  template <typename Traits, int PrefixLen, typename RecordFn>
  static void scan_last_hop(const float* close_to_start,
                            const float* row_current,
                            int first_candidate,
                            int scan_end,
                            float prefix_weight,
                            float threshold,
                            const int* prefix,
                            RecordFn&& record) {
    (void) sizeof(Traits);

    const int lanes = static_cast<int>(svcntw());
    alignas(64) float totals[kMaxFloatLanes]{};
    alignas(64) std::uint32_t hit_flags[kMaxFloatLanes]{};

    const svfloat32_t prefix_v = svdup_n_f32(prefix_weight - threshold);
    const svuint32_t one_v = svdup_n_u32(1u);
    const svuint32_t zero_u32_v = svdup_n_u32(0u);
    const svbool_t all_pg = svptrue_b32();

    auto scan_block = [&](int j, svbool_t pg, int active_lanes) {
      const svint32_t candidate_indices = svindex_s32(j, 1);

      svbool_t valid = pg;
      for (int i = 0; i < PrefixLen; ++i) {
        valid = svand_b_z(pg, valid, svcmpne_n_s32(pg, candidate_indices, prefix[i]));
      }

      const svfloat32_t row_v = svld1_f32(pg, row_current + static_cast<std::size_t>(j));
      const svfloat32_t close_v = svld1_f32(pg, close_to_start + static_cast<std::size_t>(j));
      const svfloat32_t total_v = svadd_f32_x(pg, prefix_v, svadd_f32_x(pg, row_v, close_v));
      const svbool_t hits = svcmplt_n_f32(valid, total_v, 0.0f);
      if (!svptest_any(pg, hits)) {
        return;
      }

      svst1_f32(pg, totals, total_v);
      svst1_u32(pg, hit_flags, svsel_u32(hits, one_v, zero_u32_v));

      for (int lane = 0; lane < active_lanes; ++lane) {
        if (hit_flags[lane] == 0u) {
          continue;
        }

        int path[6] = {0, 0, 0, 0, 0, 0};
        for (int i = 0; i < PrefixLen; ++i) {
          path[i] = prefix[i];
        }
        path[PrefixLen] = j + lane;
        path[PrefixLen + 1] = prefix[0];
        record(totals[lane] + threshold, PrefixLen + 1, path);
      }
    };

    int j = first_candidate;
    for (; j + lanes <= scan_end; j += lanes) {
      scan_block(j, all_pg, lanes);
    }
    if (j < scan_end) {
      const svbool_t pg = svwhilelt_b32(j, scan_end);
      const int active_lanes = static_cast<int>(svcntp_b32(all_pg, pg));
      scan_block(j, pg, active_lanes);
    }
  }
};
#endif

template <typename Detector, typename Traits, typename ScanPolicy = LooseScanPolicy>
struct SimdSearch {
  using Base = ArbitrageDetectorBase;
  using Cycle = typename Base::Cycle;
  using DenseWeights = typename Base::DenseWeights;
  using UpsertResult = typename Base::UpsertResult;

  [[nodiscard]] static std::optional<Cycle> find_best_arbitrage(
      Detector& detector,
      int max_cycle_length) {
    if (max_cycle_length < 2 || detector.n() < 2) {
      detector.cached_best_.reset();
      detector.cached_max_cycle_length_ = max_cycle_length;
      detector.cache_valid_ = true;
      return std::nullopt;
    }

    if (max_cycle_length > 5) {
      return detector.find_best_arbitrage_common(max_cycle_length);
    }

    const DenseWeights& dense = detector.dense_weights(ScanPolicy::kPadMultiple);
    const int N = dense.n;
    const int simd_end = ScanPolicy::simd_scan_end(dense);
    const std::size_t stride = static_cast<std::size_t>(dense.padded_stride);
    const auto& W = dense.weights;
    const auto& WT = dense.transpose;

    BestCandidate best;

    for (int start = 0; start < N; ++start) {
      const float* close_to_start =
          WT.data() + static_cast<std::size_t>(start) * stride;
      const float* row_start =
          W.data() + static_cast<std::size_t>(start) * stride;
      const auto& adj_start = detector.outgoing_[static_cast<std::size_t>(start)];

      int prefix[5] = {start, 0, 0, 0, 0};

      for (int a : adj_start) {
        if (a <= start) {
          continue;
        }

        prefix[1] = a;
        const float w_sa = row_start[static_cast<std::size_t>(a)];
        const float* row_a = W.data() + static_cast<std::size_t>(a) * stride;

        const float total2 = w_sa + close_to_start[static_cast<std::size_t>(a)];
        if (total2 < 0.0f) {
          const int path[3] = {start, a, start};
          record_best(best, total2, 2, path);
        }

        if (max_cycle_length >= 3) {
          ScanPolicy::template scan_last_hop<Traits, 2>(
              close_to_start,
              row_a,
              start + 1,
              simd_end,
              w_sa,
              ScanPolicy::threshold(best),
              prefix,
              [&](float total_weight, int len, const int* vertices) {
                record_best(best, total_weight, len, vertices);
              });
        }

        if (max_cycle_length >= 4) {
          const auto& adj_a = detector.outgoing_[static_cast<std::size_t>(a)];
          for (int b : adj_a) {
            if (b <= start || b == a) {
              continue;
            }

            prefix[2] = b;
            const float prefix2 = w_sa + row_a[static_cast<std::size_t>(b)];
            const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;

            ScanPolicy::template scan_last_hop<Traits, 3>(
                close_to_start,
                row_b,
                start + 1,
                simd_end,
                prefix2,
                ScanPolicy::threshold(best),
                prefix,
                [&](float total_weight, int len, const int* vertices) {
                  record_best(best, total_weight, len, vertices);
                });

            if (max_cycle_length >= 5) {
              const auto& adj_b = detector.outgoing_[static_cast<std::size_t>(b)];
              for (int c : adj_b) {
                if (c <= start || c == a || c == b) {
                  continue;
                }

                prefix[3] = c;
                const float prefix3 =
                    prefix2 +
                    W[static_cast<std::size_t>(b) * stride +
                      static_cast<std::size_t>(c)];
                const float* row_c = W.data() + static_cast<std::size_t>(c) * stride;

                ScanPolicy::template scan_last_hop<Traits, 4>(
                    close_to_start,
                    row_c,
                    start + 1,
                    simd_end,
                    prefix3,
                    ScanPolicy::threshold(best),
                    prefix,
                    [&](float total_weight, int len, const int* vertices) {
                      record_best(best, total_weight, len, vertices);
                    });
              }
            }
          }
        }
      }
    }

    if (!best.have) {
      detector.cached_best_.reset();
    } else {
      detector.cached_best_ = materialize_best(detector, best);
    }

    detector.cached_max_cycle_length_ = max_cycle_length;
    detector.cache_valid_ = true;
    return detector.cached_best_;
  }

  [[nodiscard]] static std::optional<Cycle> add_quote_and_find_best_arbitrage(
      Detector& detector,
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length) {
    const UpsertResult update =
        detector.upsert_quote(from, to, executable_rate, fee_bps);

    if (max_cycle_length < 2 || detector.n() < 2) {
      detector.cached_best_.reset();
      detector.cached_max_cycle_length_ = max_cycle_length;
      detector.cache_valid_ = true;
      return std::nullopt;
    }

    if (!detector.cache_valid_ ||
        detector.cached_max_cycle_length_ != max_cycle_length ||
        update.new_currency) {
      return find_best_arbitrage(detector, max_cycle_length);
    }

    const bool improved =
        update.new_edge ||
        update.new_weight < update.old_weight - Base::kCompareEpsilon;
    const bool worsened =
        update.existed &&
        update.new_weight > update.old_weight + Base::kCompareEpsilon;
    const bool cached_uses_edge =
        detector.cached_best_.has_value() &&
        detector.cycle_uses_edge(*detector.cached_best_, update.from, update.to);

    if (improved) {
      if (detector.cached_best_ && !cached_uses_edge && update.existed) {
        const float max_possible_improvement = update.old_weight - update.new_weight;
        if (max_possible_improvement <= Base::kCompareEpsilon) {
          return detector.cached_best_;
        }
      }

      std::optional<Cycle> through_edge =
          find_best_cycle_through_edge(detector, update.from, update.to, max_cycle_length);
      std::optional<Cycle> result;

      if (detector.cached_best_ && !cached_uses_edge) {
        result = Base::better_optional(detector.cached_best_, std::move(through_edge));
      } else {
        result = std::move(through_edge);
      }

      detector.cached_best_ = std::move(result);
      detector.cached_max_cycle_length_ = max_cycle_length;
      detector.cache_valid_ = true;
      return detector.cached_best_;
    }

    if (worsened) {
      if (!cached_uses_edge) {
        return detector.cached_best_;
      }
      return find_best_arbitrage(detector, max_cycle_length);
    }

    return detector.cached_best_;
  }

  [[nodiscard]] static std::optional<Cycle> add_book_and_find_best_arbitrage(
      Detector& detector,
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length) {
    if (!(bid > 0.0f) || !(ask > 0.0f) || bid > ask) {
      throw std::invalid_argument("invalid bid/ask");
    }
    if (!(fee_bps >= 0.0f) || fee_bps >= 10000.0f) {
      throw std::invalid_argument("fee_bps must be in [0, 10000)");
    }

    const float reverse_rate = 1.0f / ask;

    bool forward_same = false;
    bool reverse_same = false;

    const auto it_base = detector.id_by_code_.find(std::string(base));
    const auto it_quote = detector.id_by_code_.find(std::string(quote));

    if (it_base != detector.id_by_code_.end() &&
        it_quote != detector.id_by_code_.end()) {
      const int u = it_base->second;
      const int v = it_quote->second;

      const auto& forward = detector.cell(u, v);
      const auto& reverse = detector.cell(v, u);

      if (forward.exists &&
          std::fabs(forward.gross_rate - bid) <= Base::kCompareEpsilon &&
          std::fabs(forward.fee_bps - fee_bps) <= Base::kCompareEpsilon) {
        forward_same = true;
      }

      if (reverse.exists &&
          std::fabs(reverse.gross_rate - reverse_rate) <= Base::kCompareEpsilon &&
          std::fabs(reverse.fee_bps - fee_bps) <= Base::kCompareEpsilon) {
        reverse_same = true;
      }
    }

    if (forward_same && reverse_same) {
      if (detector.cache_valid_ &&
          detector.cached_max_cycle_length_ == max_cycle_length) {
        return detector.cached_best_;
      }
      return find_best_arbitrage(detector, max_cycle_length);
    }

    if (forward_same && !reverse_same) {
      return add_quote_and_find_best_arbitrage(
          detector, quote, base, reverse_rate, fee_bps, max_cycle_length);
    }

    if (!forward_same && reverse_same) {
      return add_quote_and_find_best_arbitrage(
          detector, base, quote, bid, fee_bps, max_cycle_length);
    }

    const UpsertResult forward =
        detector.upsert_quote(base, quote, bid, fee_bps);
    const UpsertResult reverse =
        detector.upsert_quote(quote, base, reverse_rate, fee_bps);

    if (max_cycle_length < 2 || detector.n() < 2) {
      detector.cached_best_.reset();
      detector.cached_max_cycle_length_ = max_cycle_length;
      detector.cache_valid_ = true;
      return std::nullopt;
    }

    if (!detector.cache_valid_ ||
        detector.cached_max_cycle_length_ != max_cycle_length ||
        forward.new_currency ||
        reverse.new_currency) {
      return find_best_arbitrage(detector, max_cycle_length);
    }

    const bool forward_improved =
        forward.new_edge ||
        forward.new_weight < forward.old_weight - Base::kCompareEpsilon;
    const bool forward_worsened =
        forward.existed &&
        forward.new_weight > forward.old_weight + Base::kCompareEpsilon;
    const bool reverse_improved =
        reverse.new_edge ||
        reverse.new_weight < reverse.old_weight - Base::kCompareEpsilon;
    const bool reverse_worsened =
        reverse.existed &&
        reverse.new_weight > reverse.old_weight + Base::kCompareEpsilon;

    const bool cached_uses_forward =
        detector.cached_best_.has_value() &&
        detector.cycle_uses_edge(*detector.cached_best_, forward.from, forward.to);
    const bool cached_uses_reverse =
        detector.cached_best_.has_value() &&
        detector.cycle_uses_edge(*detector.cached_best_, reverse.from, reverse.to);

    if ((forward_worsened && cached_uses_forward) ||
        (reverse_worsened && cached_uses_reverse)) {
      return find_best_arbitrage(detector, max_cycle_length);
    }

    if (!forward_improved && !reverse_improved) {
      return detector.cached_best_;
    }

    std::optional<Cycle> result;
    const bool cached_uses_improved_edge =
        (forward_improved && cached_uses_forward) ||
        (reverse_improved && cached_uses_reverse);

    if (detector.cached_best_ && !cached_uses_improved_edge) {
      result = detector.cached_best_;
    }

    if (forward_improved) {
      result = Base::better_optional(
          std::move(result),
          find_best_cycle_through_edge(
              detector, forward.from, forward.to, max_cycle_length));
    }

    if (reverse_improved) {
      result = Base::better_optional(
          std::move(result),
          find_best_cycle_through_edge(
              detector, reverse.from, reverse.to, max_cycle_length));
    }

    detector.cached_best_ = std::move(result);
    detector.cached_max_cycle_length_ = max_cycle_length;
    detector.cache_valid_ = true;
    return detector.cached_best_;
  }

  [[nodiscard]] static std::vector<Cycle> find_arbitrage(
      Detector& detector,
      int max_cycle_length) {
    if (max_cycle_length < 2 || detector.n() < 2) {
      return {};
    }

    if (max_cycle_length > 5) {
      return detector.find_arbitrage_common(max_cycle_length);
    }

    const DenseWeights& dense = detector.dense_weights(ScanPolicy::kPadMultiple);
    const int N = dense.n;
    const int simd_end = ScanPolicy::simd_scan_end(dense);
    const std::size_t stride = static_cast<std::size_t>(dense.padded_stride);
    const auto& W = dense.weights;
    const auto& WT = dense.transpose;

    std::vector<Cycle> cycles;
    cycles.reserve(512);

    for (int start = 0; start < N; ++start) {
      const float* close_to_start =
          WT.data() + static_cast<std::size_t>(start) * stride;
      const float* row_start =
          W.data() + static_cast<std::size_t>(start) * stride;
      const auto& adj_start = detector.outgoing_[static_cast<std::size_t>(start)];

      int prefix[5] = {start, 0, 0, 0, 0};

      for (int a : adj_start) {
        if (a <= start) {
          continue;
        }

        prefix[1] = a;
        const float w_sa = row_start[static_cast<std::size_t>(a)];
        const float* row_a = W.data() + static_cast<std::size_t>(a) * stride;

        const float total2 = w_sa + close_to_start[static_cast<std::size_t>(a)];
        if (total2 < 0.0f) {
          const int path[3] = {start, a, start};
          cycles.push_back(materialize(detector, total2, 2, path));
        }

        if (max_cycle_length >= 3) {
          ScanPolicy::template scan_last_hop<Traits, 2>(
              close_to_start,
              row_a,
              start + 1,
              simd_end,
              w_sa,
              0.0f,
              prefix,
              [&](float total_weight, int len, const int* vertices) {
                cycles.push_back(materialize(detector, total_weight, len, vertices));
              });
        }

        if (max_cycle_length >= 4) {
          const auto& adj_a = detector.outgoing_[static_cast<std::size_t>(a)];
          for (int b : adj_a) {
            if (b <= start || b == a) {
              continue;
            }

            prefix[2] = b;
            const float prefix2 = w_sa + row_a[static_cast<std::size_t>(b)];
            const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;

            ScanPolicy::template scan_last_hop<Traits, 3>(
                close_to_start,
                row_b,
                start + 1,
                simd_end,
                prefix2,
                0.0f,
                prefix,
                [&](float total_weight, int len, const int* vertices) {
                  cycles.push_back(materialize(detector, total_weight, len, vertices));
                });

            if (max_cycle_length >= 5) {
              const auto& adj_b = detector.outgoing_[static_cast<std::size_t>(b)];
              for (int c : adj_b) {
                if (c <= start || c == a || c == b) {
                  continue;
                }

                prefix[3] = c;
                const float prefix3 =
                    prefix2 +
                    W[static_cast<std::size_t>(b) * stride +
                      static_cast<std::size_t>(c)];
                const float* row_c = W.data() + static_cast<std::size_t>(c) * stride;

                ScanPolicy::template scan_last_hop<Traits, 4>(
                    close_to_start,
                    row_c,
                    start + 1,
                    simd_end,
                    prefix3,
                    0.0f,
                    prefix,
                    [&](float total_weight, int len, const int* vertices) {
                      cycles.push_back(materialize(detector, total_weight, len, vertices));
                    });
              }
            }
          }
        }
      }
    }

    std::sort(cycles.begin(), cycles.end(), Base::is_better_cycle);
    return cycles;
  }

  [[nodiscard]] static std::vector<Cycle> add_quote_and_find_arbitrage(
      Detector& detector,
      std::string_view from,
      std::string_view to,
      float executable_rate,
      float fee_bps,
      int max_cycle_length) {
    (void) detector.upsert_quote(from, to, executable_rate, fee_bps);
    detector.invalidate_cache();
    return find_arbitrage(detector, max_cycle_length);
  }

  [[nodiscard]] static std::vector<Cycle> add_book_and_find_arbitrage(
      Detector& detector,
      std::string_view base,
      std::string_view quote,
      float bid,
      float ask,
      float fee_bps,
      int max_cycle_length) {
    if (!(bid > 0.0f) || !(ask > 0.0f) || bid > ask) {
      throw std::invalid_argument("invalid bid/ask");
    }
    if (!(fee_bps >= 0.0f) || fee_bps >= 10000.0f) {
      throw std::invalid_argument("fee_bps must be in [0, 10000)");
    }

    (void) detector.upsert_quote(base, quote, bid, fee_bps);
    (void) detector.upsert_quote(quote, base, 1.0f / ask, fee_bps);
    detector.invalidate_cache();
    return find_arbitrage(detector, max_cycle_length);
  }

private:
  struct BestCandidate {
    bool have{false};
    float weight{std::numeric_limits<float>::infinity()};
    int len{0};
    int vertices[6] = {0, 0, 0, 0, 0, 0};
  };

  static void record_best(BestCandidate& best,
                          float total_weight,
                          int len,
                          const int* vertices) {
    if (!(total_weight < 0.0f)) {
      return;
    }

    bool better = false;
    if (!best.have) {
      better = true;
    } else if (total_weight < best.weight - Base::kCompareEpsilon) {
      better = true;
    } else if (!(best.weight < total_weight - Base::kCompareEpsilon)) {
      if (len < best.len) {
        better = true;
      } else if (
          len == best.len &&
          std::lexicographical_compare(vertices,
                                       vertices + len + 1,
                                       best.vertices,
                                       best.vertices + best.len + 1)) {
        better = true;
      }
    }

    if (better) {
      best.have = true;
      best.weight = total_weight;
      best.len = len;
      for (int i = 0; i <= len; ++i) {
        best.vertices[i] = vertices[i];
      }
    }
  }

  [[nodiscard]] static Cycle materialize(Detector& detector,
                                         float total_weight,
                                         int len,
                                         const int* vertices) {
    std::vector<int> path(static_cast<std::size_t>(len) + 1);
    for (int i = 0; i <= len; ++i) {
      path[static_cast<std::size_t>(i)] = vertices[i];
    }

    float gain_factor = 1.0f;
    for (int i = 0; i < len; ++i) {
      gain_factor *= detector.cell(vertices[i], vertices[i + 1]).net_rate;
    }

    return detector.materialize_cycle(path, total_weight, gain_factor);
  }

  [[nodiscard]] static Cycle materialize_best(Detector& detector,
                                              const BestCandidate& best) {
    return materialize(detector, best.weight, best.len, best.vertices);
  }

  [[nodiscard]] static std::optional<Cycle> find_best_cycle_through_edge(
      Detector& detector,
      int from,
      int to,
      int max_cycle_length) {
    if (max_cycle_length < 2 || from == to) {
      return std::nullopt;
    }

    if (max_cycle_length > 5) {
      return detector.find_best_cycle_through_edge_scalar(from, to, max_cycle_length);
    }

    const DenseWeights& dense = detector.dense_weights(ScanPolicy::kPadMultiple);
    const int simd_end = ScanPolicy::simd_scan_end(dense);
    const std::size_t stride = static_cast<std::size_t>(dense.padded_stride);
    const auto& W = dense.weights;
    const auto& WT = dense.transpose;

    const float first_w =
        W[static_cast<std::size_t>(from) * stride + static_cast<std::size_t>(to)];
    if (!std::isfinite(first_w)) {
      return std::nullopt;
    }

    const float* close_to_start =
        WT.data() + static_cast<std::size_t>(from) * stride;
    const float* row_to = W.data() + static_cast<std::size_t>(to) * stride;

    BestCandidate best;
    int prefix[5] = {from, to, 0, 0, 0};

    const float total2 = first_w + close_to_start[static_cast<std::size_t>(to)];
    if (total2 < 0.0f) {
      const int path[3] = {from, to, from};
      record_best(best, total2, 2, path);
    }

    if (max_cycle_length >= 3) {
      ScanPolicy::template scan_last_hop<Traits, 2>(
          close_to_start,
          row_to,
          0,
          simd_end,
          first_w,
          ScanPolicy::threshold(best),
          prefix,
          [&](float total_weight, int len, const int* vertices) {
            record_best(best, total_weight, len, vertices);
          });
    }

    if (max_cycle_length >= 4) {
      const auto& adj_to = detector.outgoing_[static_cast<std::size_t>(to)];
      for (int b : adj_to) {
        if (b == from || b == to) {
          continue;
        }

        prefix[2] = b;
        const float prefix2 = first_w + row_to[static_cast<std::size_t>(b)];
        const float* row_b = W.data() + static_cast<std::size_t>(b) * stride;

        ScanPolicy::template scan_last_hop<Traits, 3>(
            close_to_start,
            row_b,
            0,
            simd_end,
            prefix2,
            ScanPolicy::threshold(best),
            prefix,
            [&](float total_weight, int len, const int* vertices) {
              record_best(best, total_weight, len, vertices);
            });

        if (max_cycle_length >= 5) {
          const auto& adj_b = detector.outgoing_[static_cast<std::size_t>(b)];
          for (int c : adj_b) {
            if (c == from || c == to || c == b) {
              continue;
            }

            prefix[3] = c;
            const float prefix3 =
                prefix2 +
                W[static_cast<std::size_t>(b) * stride +
                  static_cast<std::size_t>(c)];
            const float* row_c = W.data() + static_cast<std::size_t>(c) * stride;

            ScanPolicy::template scan_last_hop<Traits, 4>(
                close_to_start,
                row_c,
                0,
                simd_end,
                prefix3,
                ScanPolicy::threshold(best),
                prefix,
                [&](float total_weight, int len, const int* vertices) {
                  record_best(best, total_weight, len, vertices);
                });
          }
        }
      }
    }

    if (!best.have) {
      return std::nullopt;
    }
    return materialize_best(detector, best);
  }
};

template <typename Detector, typename Traits, typename ScanPolicy = LooseScanPolicy>
using X86SimdSearch = SimdSearch<Detector, Traits, ScanPolicy>;

} // namespace negcycle
