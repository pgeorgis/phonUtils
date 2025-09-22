import os
import re
import sys
from functools import lru_cache
from itertools import combinations
from typing import Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonUtils.constants import (ALVEOLARS, ALVEOLOPALATALS, BILABIALS,
                                 DENTALS, EPIGLOTTALS, FRONT_VOWEL_REGEX,
                                 GLOTTALS, LABIODENTALS, LATERALS, LIQUIDS,
                                 NASAL_REGEX, PALATALS, PHARYNGEALS,
                                 POSTALVEOLARS, RETROFLEXES, RHOTIC_REGEX,
                                 UVULARS, VELARS)
from phonUtils.segment import Segment
from phonUtils.sonority import relative_sonority
from phonUtils.syllables import syllabify

# CONSTANTS
SEGMENT_CH = 'S'
TONEME_CH = 'T'
PHON_ENV_SPLIT_CH = "|"
PHON_ENV_SEP = "_"
BASE_SEGMENT_ENV = f'{PHON_ENV_SPLIT_CH}{SEGMENT_CH}{PHON_ENV_SPLIT_CH}'
BASE_TONEME_ENV = f'{PHON_ENV_SPLIT_CH}{TONEME_CH}{PHON_ENV_SPLIT_CH}'
OPEN_SYLLABLE_ENV = 'SylOpen'
CLOSED_SYLLABLE_ENV = 'SylClosed'
SYLLABLE_ONSET_ENV = 'SylOnset'
SYLLABLE_CODA_ENV = 'SylCoda'
BOUNDARY_TOKEN = "#"
PHON_ENV_REGEX = re.compile(rf'.*{re.escape(PHON_ENV_SPLIT_CH)}[{SEGMENT_CH}{TONEME_CH}]{re.escape(PHON_ENV_SPLIT_CH)}.*')
PHON_ENV_WITH_AFFIX_REGEXES = [
    re.compile(rf'.+{re.escape(PHON_ENV_SPLIT_CH)}{SEGMENT_CH}{re.escape(PHON_ENV_SPLIT_CH)}.*'),
    re.compile(rf'.*{re.escape(PHON_ENV_SPLIT_CH)}{SEGMENT_CH}{re.escape(PHON_ENV_SPLIT_CH)}.+'),
]

PHON_ENV_MAP = {
    "FRONT": {
        "symbol": "F",
        "regex": FRONT_VOWEL_REGEX,
        "match_chs": {'j', 'ɥ'},
    },
    "NASAL": {
        "symbol": "N",
        "regex": NASAL_REGEX,
    },
    "RHOTIC": {
        "symbol": "R",
        "regex": RHOTIC_REGEX,
    },
    "LATERAL": {
        "symbol": "L",
        "regex": re.compile(f"[{''.join(LATERALS)}ˡ]"),
    },
    "LIQUID": {
        "symbol": "RL",
        "regex": re.compile(f"[{''.join(LIQUIDS)}]"),
    },
    "LABIAL": {
        "symbol": "B",
        "regex": re.compile(f"[{''.join(BILABIALS.union(LABIODENTALS))}ʷᵝ]"),
    },
    "DENTAL/ALVEOLAR": {
        "symbol": "D",
        "regex": re.compile(f"[{''.join(DENTALS.union(ALVEOLARS))}]"),
    },
    "POST-ALVEOLAR": {
        "symbol": "Š",
        "regex": re.compile(f"[{''.join(POSTALVEOLARS.union(RETROFLEXES))}]"),
    },
    "PALATAL": {
        "symbol": "P",
        "regex": re.compile(f"[{''.join(PALATALS.union(ALVEOLOPALATALS))}]"),
    },
    "VELAR/UVULAR": {
        "symbol": "K",
        "regex": re.compile(f"[{''.join(VELARS.union(UVULARS))}ˠ]"),
    },
    "PHARYNGEAL/(EPI)GLOTTAL": {
        "symbol": "H",
        "regex": re.compile(f"[{''.join(GLOTTALS.union(EPIGLOTTALS).union(PHARYNGEALS))}ˤˀ]"),
    },
    "VOICELESS": {
        "symbol": "-Voice",
        "features": {"periodicGlottalSource": 0},
    },
    "VOICED": {
        "symbol": "+Voice",
        "features": {"periodicGlottalSource": 1},
    },
    "VOWEL": {
        "symbol": "V",
        "phone_class": ['VOWEL', 'DIPHTHONG'],
    },
    "CONSONANT": {
        "symbol": "C",
        "phone_class": ['CONSONANT', 'GLIDE'],
    },
    "ACCENTED": {
        "symbol": "A",
        "phone_class": ['TONEME', 'SUPRASEGMENTAL'],
    },
}
ALL_PHON_ENVS = list(PHON_ENV_MAP.keys())


class PhonEnv:
    def __init__(self, segments, i, gap_ch=None, phon_env_map=PHON_ENV_MAP, **kwargs):
        self.phon_env_map = phon_env_map
        self.gap_ch = gap_ch
        if self.gap_ch:
            assert isinstance(self.gap_ch, str)
            i, segments = self.preprocess_aligned_sequence(segments, i)
        self.index = i
        self.segment_i = None
        self.supra_segs = [Segment(s) if not self.is_gappy(s) else s for s in segments]
        self.segments, self.adjust_n = self.sep_segs_from_suprasegs(self.supra_segs, self.index)
        self.adjusted_index = self.index - self.adjust_n
        self.gappy = self.is_gappy(self.segment_i)
        self.syllables = self.get_syllables()
        self.phon_env = self.get_phon_env(**kwargs)

    def preprocess_aligned_sequence(self, segments, i):
        """Drop gaps and boundaries and flatten complex ngrams."""
        minus_offset, plus_offset = 0, 0
        adj_segments = []
        segments = [self.preprocess_ngram_segment(seg) for seg in segments]
        for segment in segments[:i]:
            if isinstance(segment, str) and BOUNDARY_TOKEN in segment:
                minus_offset += 1
                continue
            elif isinstance(segment, tuple) and BOUNDARY_TOKEN in segment[0]:
                adj_segments.extend(segment[1:])
                plus_offset += len(segment) - 2
                continue
            if segment == self.gap_ch:
                minus_offset += 1
            else:
                if isinstance(segment, tuple):
                    adj_segments.extend(segment)
                    plus_offset += len(segment) - 1
                else:
                    adj_segments.append(segment)
        adjusted_i = i - minus_offset + plus_offset
        adj_segments.append(segments[i])
        for segment in segments[i:][1:]:
            if not self.is_gappy(segment):
                if isinstance(segment, tuple) and any(self.is_gappy(subseg) for subseg in segment):
                    if BOUNDARY_TOKEN in segment[-1]:
                        adj_segments.extend(segment[:-1])
                    else:
                        assert self.gap_ch in segment
                        adj_segments.extend([subseg for subseg in segment if subseg != self.gap_ch])
                elif isinstance(segment, tuple):
                    adj_segments.extend(segment)
                else:
                    adj_segments.append(segment)
        return adjusted_i, adj_segments

    def sep_segs_from_suprasegs(self, segments, i):
        adjust_n = 0
        segs = []
        for j, seg in enumerate(segments):
            if isinstance(seg, Segment) and seg.phone_class not in ('TONEME', 'SUPRASEGMENTAL'):
                segs.append(seg)
            elif isinstance(seg, str): # str: gap or boundary
                segs.append(seg)
            elif j < i:
                adjust_n += 1
            if j == i:
                self.segment_i = seg
        return segs, adjust_n

    def get_syllables(self):
        """Return syllable dictionary with segment indices constituting syllable nuclei as keys and Syllable objects as values."""
        segments = []
        for seg in self.segments:
            if isinstance(seg, Segment):
                segments.append(seg.segment)
            else:
                # skip gaps
                assert self.is_gappy(seg)
        return syllabify(''.join(segments))

    def get_phon_env(self):
        """Returns a string representing the phonological environment of a segment within a word"""

        # Tonemes/suprasegmentals
        if isinstance(self.segment_i, Segment) and self.segment_i.phone_class in ('TONEME', 'SUPRASEGMENTAL'):
            return BASE_TONEME_ENV

        # Word-initial segments (free-standing segments also considered word-initial)
        env = BASE_SEGMENT_ENV
        i = self.adjusted_index
        if i == 0:
            if len(self.segments) > 1:
                next_segment = self.segments[i+1]
            else:
                next_segment = None

            # Word-initial segments: #S
            if next_segment:
                env = BOUNDARY_TOKEN + env

                # Add relative sonority environment for non-gaps
                if not self.gappy:
                    _, post_sonority = self.relative_sonority(next_seg=next_segment)
                    env += PHON_ENV_SEP + post_sonority

                # Add feature environments
                env = self.add_envs(env, next_segment, suffix=True)

                # # Add the next segment itself
                # env += PHON_ENV_SEP + next_segment.segment

                # If vowel, mark whether syllable is open or closed
                env = self.add_syllable_env(i, env)

                return env

            # Free-standing segments
            else:
                if self.segment_i.phone_class in {"VOWEL", "DIPHTHONG"}:
                    return f"{BOUNDARY_TOKEN}{BASE_SEGMENT_ENV}{OPEN_SYLLABLE_ENV}{BOUNDARY_TOKEN}"
                return f"{BOUNDARY_TOKEN}{BASE_SEGMENT_ENV}{BOUNDARY_TOKEN}"

        # Word-final segments: S#
        elif i == len(self.segments)-1:
            assert len(self.segments) > 1
            prev_segment = self.segments[i-1]
            env += BOUNDARY_TOKEN

            # Add relative sonority environment for non-gaps
            if not self.gappy:
                pre_sonority, _ = self.relative_sonority(prev_seg=prev_segment)
                env = PHON_ENV_SEP.join([pre_sonority, env])

            # Add feature environments
            env = self.add_envs(env, prev_segment, prefix=True)

            # # Add the previous segment itself
            # env = PHON_ENV_SEP.join([prev_segment.segment, env])

            # If vowel, mark whether syllable is open or closed
            env = self.add_syllable_env(i, env)

            return env

        # Word-medial segments
        else:
            prev_segment, next_segment = self.segments[i-1], self.segments[i+1]

            # Add sonority environments for non-gaps
            if not self.gappy:
                prev_sonority, post_sonority = self.relative_sonority(prev_seg=prev_segment, next_seg=next_segment)
                env = PHON_ENV_SEP.join([prev_sonority, env, post_sonority])

            # Add feature environments
            env = self.add_envs(env, prev_segment, prefix=True)
            env = self.add_envs(env, next_segment, suffix=True)

            # # Add the next segment itself
            # env += PHON_ENV_SEP + next_segment.segment

            # # Add the previous segment itself
            # env = PHON_ENV_SEP.join([prev_segment.segment, env])

            # If vowel, mark whether syllable is open or closed
            env = self.add_syllable_env(i, env)

            return env
    
    @staticmethod
    def preprocess_ngram_segment(ngram_seg) -> str | tuple[str, ...]:
        if isinstance(ngram_seg, tuple) and len(ngram_seg) == 1:
            ngram_seg = ngram_seg[0]
            assert isinstance(ngram_seg, str)
        elif isinstance(ngram_seg, str):
            pass
        else:
            assert isinstance(ngram_seg, tuple) and len(ngram_seg) > 1
        return ngram_seg

    def is_gappy(self, seg):
        if isinstance(seg, Segment):
            # segments cannot be gappy
            return False
        seg = self.preprocess_ngram_segment(seg)
        return isinstance(seg, str) and (seg == self.gap_ch or BOUNDARY_TOKEN in seg)

    def relative_sonority(self, prev_seg=None, next_seg=None):
        return relative_sonority(self.segment_i, prev_seg=prev_seg, next_seg=next_seg)

    @staticmethod
    def env_matches(segment: Segment,
                    regex: re.Pattern = None,
                    match_chs: Iterable = None,
                    features: dict = None
                    ) -> bool:
        if features:
            if all(segment.features.get(feature) == feature_val for feature, feature_val in features.items()):
                return True
            return False
        if regex and regex.search(segment.segment):
            return True
        if match_chs and segment.base in match_chs:
            return True
        return False

    def add_env(self,
                env,
                segment,
                symbol,
                regex=None,
                match_chs=None,
                phone_class=None,
                features=None,
                prefix=None,
                suffix=None,
                sep=PHON_ENV_SEP):
        assert prefix is not None or suffix is not None
        if phone_class and segment.phone_class in phone_class:
            pass
        elif self.env_matches(segment=segment, features=features, regex=regex, match_chs=match_chs):
            pass
        else:
            return env
        if prefix:
            return symbol + sep + env
        else: # suffix
            return env + sep + symbol

    def add_envs(self, env, segment, **kwargs):
        for _, encoding_map in self.phon_env_map.items():
            if encoding_map.get("active", True) is False:
                continue
            symbol = encoding_map["symbol"]
            regex = encoding_map.get("regex", None)
            match_chs = encoding_map.get("match_chs", None)
            phone_class = encoding_map.get("phone_class", None)
            features = encoding_map.get("features", None)
            env = self.add_env(
                env,
                segment,
                symbol=symbol,
                regex=regex,
                match_chs=match_chs,
                phone_class=phone_class,
                features=features,
                **kwargs
            )
        return env 

    def add_syllable_env(self, i, env):
        """Add 'OPEN' or 'CLOSED' to syllable nuclei, 'ONSET' or 'CODA' to other segments within a syllable."""
        if i in self.syllables:
            syllable_type = self.syllables[i].type
            assert syllable_type in {"OPEN", "CLOSED", "OTHER"}
            if syllable_type == "OPEN":
                syllable_env = OPEN_SYLLABLE_ENV
            elif syllable_type == "CLOSED":
                syllable_env = CLOSED_SYLLABLE_ENV
            else: # "OTHER"
                return env
        else:
            syllable_env = SYLLABLE_CODA_ENV
            for nucleus_i in self.syllables:
                if i < nucleus_i:
                    syllable_env = SYLLABLE_ONSET_ENV
                    break
        if syllable_env == SYLLABLE_ONSET_ENV:
            env = PHON_ENV_SEP.join([syllable_env, env])
        else:
            env += PHON_ENV_SEP + syllable_env
        return env

    def ngrams(self, exclude=set()):
        """Returns set of phonological environment strings of equal and lower order,
        e.g. ">|S|#" -> ">|S", "S|#", ">|S|#"

        Returns:
            set: possible equal and lower order phonological environment strings
        """
        return list_phon_env_ngram_subcontexts(self.phon_env, exclude=exclude)

    def __str__(self):
        return self.phon_env


@lru_cache
def list_phon_env_ngram_subcontexts(phonEnv, exclude_base=True):
    """Returns set of phonological environment strings of equal and lower order,
    e.g. ">|S|#" -> ">|S", "S|#", ">|S|#"

    Returns:
        set: possible equal and lower order phonological environment strings
    """
    if any(regex.search(phonEnv) for regex in PHON_ENV_WITH_AFFIX_REGEXES):
        prefix, base, suffix = phonEnv.split(PHON_ENV_SPLIT_CH)
        prefix = [p for p in prefix.split(PHON_ENV_SEP) if p]
        prefixes = set()
        for i in range(1, len(prefix)+1):
            for x in combinations(prefix, i):
                prefixes.add(PHON_ENV_SEP.join(x))
        prefixes.add('')
        suffix = [s for s in suffix.split(PHON_ENV_SEP) if s]
        suffixes = set()
        for i in range(1, len(suffix)+1):
            for x in combinations(suffix, i):
                suffixes.add(PHON_ENV_SEP.join(x))
        suffixes.add('')
        ngrams = set()
        for prefix in prefixes:
            for suffix in suffixes:
                ngrams.add(f'{prefix}{PHON_ENV_SPLIT_CH}{SEGMENT_CH}{PHON_ENV_SPLIT_CH}{suffix}')
    else:
        assert phonEnv in (BASE_TONEME_ENV, BASE_SEGMENT_ENV)
        ngrams = [phonEnv]

    if exclude_base:
        return [ngram for ngram in ngrams if ngram != f'{PHON_ENV_SPLIT_CH}{SEGMENT_CH}{PHON_ENV_SPLIT_CH}']
    return ngrams


def custom_phon_env_map(active_envs: list) -> dict:
    """Adjust the default phoneEnv map to activate only the specified environments."""
    phon_env_map = PHON_ENV_MAP.copy()
    for env in phon_env_map:
        if env not in active_envs:
            phon_env_map[env]["active"] = False
    for env in active_envs:
        if env not in phon_env_map:
            raise ValueError(f"Unknown phonEnv type '{env}'")
    return phon_env_map


def get_phon_env(segments, i, active_envs=ALL_PHON_ENVS, **kwargs):
    phon_env_map = custom_phon_env_map(active_envs)
    phon_env = PhonEnv(segments, i, phon_env_map=phon_env_map, **kwargs)
    return phon_env.phon_env
