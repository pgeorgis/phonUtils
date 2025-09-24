import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from segment import Segment

IDENTICAL_SEGMENT_CH = "S"
IDENTICAL_SONORITY_CH = "="
INCREASING_SONORITY_CH = "<"
DECREASING_SONORITY_CH = ">"

# Adapted from Parker's (2002) universal sonority hierarchy
# See figure: https://www.researchgate.net/publication/336652515/figure/fig1/AS:815405140561923@1571419143959/Adapted-version-of-Parkers-2002-sonority-hierarchy.ppm
SONORITY_LEVELS = {
    "SUPRASEGMENTAL": 17,
    "TONEME": 17,
    # Vowels
    "OPEN VOWELS": 16,
    "NEAR-OPEN VOWELS": 16,
    "OPEN-MID VOWELS": 15,
    "MID VOWELS": 15,
    "CLOSE-MID VOWELS": 15,
    "NEAR-CLOSE VOWELS": 14,
    "CLOSE VOWELS": 14,
    "/ə/": 13,
    "/ɨ/": 12,
    # Sonorant consonants
    "GLIDES": 11,
    "GENERAL APPROXIMANTS": 10,
    "LATERAL APPROXIMANTS": 9,
    "TAPS": 8,
    "FLAPS": 8,
    "TRILLS": 7,
    "NASALS": 6,
    # Fricatives
    "/h/": 5,
    "VOICED FRICATIVES": 4,
    "VOICELESS FRICATIVES": 3,
    # Affricates, plosives, implosives, clicks
    "VOICED PLOSIVES": 2,
    "VOICED AFFRICATES": 2,
    "VOICED IMPLOSIVES": 2,
    "VOICED CLICKS": 2,
    "VOICELESS PLOSIVES": 1,
    "VOICELESS AFFRICATES": 1,
    "VOICELESS IMPLOSIVES": 1,
    "VOICELESS CLICKS": 1,
}


def relative_prev_sonority(seg: Segment,
                           prev_seg: Segment
                           ) -> str:
    """Retrieve the sonority of a segment relative to the preceding segment."""
    if prev_seg == seg.sonority:
        return IDENTICAL_SEGMENT_CH
    elif prev_seg.sonority == seg.sonority:
        return IDENTICAL_SONORITY_CH
    elif prev_seg.sonority < seg.sonority:
        return INCREASING_SONORITY_CH
    else: # prev_sonority > sonority_i
        assert prev_seg.sonority > seg.sonority
        return DECREASING_SONORITY_CH


def relative_post_sonority(seg: Segment,
                           next_seg: Segment
                           ) -> str:
    """Retrieve the sonority of a segment relative to the following segment."""
    if next_seg.segment == seg.segment:
            return IDENTICAL_SEGMENT_CH
    elif next_seg.sonority == seg.sonority:
        return IDENTICAL_SONORITY_CH
    elif next_seg.sonority < seg.sonority:
        return DECREASING_SONORITY_CH
    else: # sonority_i > next_seg
        assert next_seg.sonority > seg.sonority
        return INCREASING_SONORITY_CH


def relative_sonority(seg: Segment,
                      prev_seg: Segment = None,
                      next_seg: Segment = None
                      ) -> tuple[str, str]:
    """Return a segment's sonority relative to the preceding and/or following segments."""
    assert prev_seg is not None or next_seg is not None
    if prev_seg is None:
        prev_son = None
    else:
        prev_son = relative_prev_sonority(seg, prev_seg)
    if next_seg is None:
        post_son = None
    else:
        post_son = relative_post_sonority(seg, next_seg)
    return prev_son, post_son
