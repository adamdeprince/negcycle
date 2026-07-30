from types import SimpleNamespace

import negcycle


def compiled_records(*kinds):
    return [
        SimpleNamespace(
            kind=kind,
            name=kind.name,
            compiled=True,
            available=False,
        )
        for kind in kinds
    ]


def test_avx512_is_preferred_when_avx2_and_avx512_are_available(monkeypatch):
    kinds = (
        negcycle._BackendKind.generic,
        negcycle._BackendKind.x86_avx2,
        negcycle._BackendKind.x86_avx512,
    )
    monkeypatch.setattr(
        negcycle,
        "_compiled_backends",
        lambda: compiled_records(*kinds),
    )
    monkeypatch.setattr(
        negcycle,
        "_detected_feature_names",
        lambda: frozenset({"X86_AVX2", "X86_AVX512F"}),
    )

    assert negcycle.detect_best_backend() == negcycle._BackendKind.x86_avx512


def test_avx2_remains_the_fallback_without_avx512(monkeypatch):
    kinds = (
        negcycle._BackendKind.generic,
        negcycle._BackendKind.x86_avx2,
        negcycle._BackendKind.x86_avx512,
    )
    monkeypatch.setattr(
        negcycle,
        "_compiled_backends",
        lambda: compiled_records(*kinds),
    )
    monkeypatch.setattr(
        negcycle,
        "_detected_feature_names",
        lambda: frozenset({"X86_AVX2"}),
    )

    assert negcycle.detect_best_backend() == negcycle._BackendKind.x86_avx2
