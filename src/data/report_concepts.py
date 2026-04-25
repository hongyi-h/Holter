"""Report concept extraction from Chinese Holter conclusions."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ConceptHit:
    name: str
    name_zh: str
    matched: str


CONCEPT_PATTERNS: list[tuple[str, str, str]] = [
    # (concept_name, chinese_name, regex_pattern)
    ("sinus_rhythm", "窦性心律", r"窦性心律"),
    ("sinus_bradycardia", "窦性心动过缓", r"窦性心动过缓"),
    ("sinus_tachycardia", "窦性心动过速", r"窦性心动过速"),
    ("sinus_arrhythmia", "窦性心律不齐", r"窦[性]?[心律]?不齐"),
    ("frequent_pvc", "频发室早", r"频发室[性早搏]|室[性]?早搏[：:]*[^\d]*\d{3,}"),
    ("pvc_bigeminy", "室早二联律", r"室早二联律|二联律"),
    ("pvc_trigeminy", "室早三联律", r"室早三联律|三联律"),
    ("pvc_couplet", "室早成对", r"成对室早|室早成对|成对"),
    ("pvc_run", "短阵室速", r"短阵室速|室性心动过速|室速"),
    ("pvc_present", "室性早搏", r"室[性]?早搏"),
    ("frequent_pac", "频发室上早", r"频发室上[性早搏]|房[性]?早搏[：:]*[^\d]*\d{3,}"),
    ("pac_present", "室上性早搏", r"室上[性]?早搏|房[性]?早搏"),
    ("svt_run", "短阵室上速", r"短阵室上速|室上性心动过速|阵发性室上"),
    ("atrial_fibrillation", "房颤/房扑", r"房颤|心房颤动|房扑|心房扑动"),
    ("pause_long", "长间歇", r"长R-?R间期|最长R-?R|间歇[>≥]\s*[2-9]|停搏"),
    ("av_block", "房室传导阻滞", r"房室传导阻滞|[Ⅰ-Ⅲ]度.*阻滞|AVB"),
    ("bundle_branch_block", "束支传导阻滞", r"束支[传导]?阻滞|右束支|左束支|RBBB|LBBB"),
    ("st_t_change", "ST-T改变", r"ST-?T[改变异常]|ST段[压低抬高改变]|缺血型"),
    ("st_t_normal", "ST-T未见异常", r"未见缺血型ST-?T|ST-?T未见[明显异常]"),
]


class ReportConceptExtractor:
    def __init__(self, patterns: list[tuple[str, str, str]] | None = None):
        self.patterns = patterns or CONCEPT_PATTERNS
        self._compiled = [(n, zh, re.compile(p)) for n, zh, p in self.patterns]

    @property
    def concept_names(self) -> list[str]:
        return [n for n, _, _ in self.patterns]

    @property
    def n_concepts(self) -> int:
        return len(self.patterns)

    def extract(self, conclusion: str) -> dict[str, bool]:
        result = {}
        for name, zh, pat in self._compiled:
            result[name] = bool(pat.search(conclusion))
        return result

    def extract_vector(self, conclusion: str) -> list[int]:
        labels = self.extract(conclusion)
        return [int(labels[n]) for n in self.concept_names]

    def extract_hits(self, conclusion: str) -> list[ConceptHit]:
        hits = []
        for name, zh, pat in self._compiled:
            m = pat.search(conclusion)
            if m:
                hits.append(ConceptHit(name=name, name_zh=zh, matched=m.group()))
        return hits
