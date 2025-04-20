import os
import re
import sys
from typing import Self, Optional

import cython
from itertools import combinations

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load phonological constants initialized in initPhoneData.py
from phonUtils.initPhoneData import (front_vowel_regex, nasal_regex,
                                     rhotic_regex)
from phonUtils.segment import Segment, _toSegment
from phonUtils.syllables import syllabify

# CONSTANTS
PHON_ENV_REGEX = re.compile(r'.*\|[ST]\|.*')
BASE_SEGMENT_ENV = "|S|"
BASE_TONEME_ENV = "|T|"
BOUNDARY_TOKEN = "#"

# HELPER FUNCTIONS
@cython.cfunc
@cython.returns(cython.bint)
@cython.locals(
    ch = object,
    regex = object,
    ch_list = set,
)
def _is_env(ch: Segment,
            regex: re.Pattern[str] = None,
            ch_list: set=None) -> cython.bint:
    if regex and regex.search(ch):
        return True
    if ch_list and ch in ch_list:
        return True
    return False

# Relative sonority functions
@cython.cfunc
@cython.returns(str)
def relative_prev_sonority(seg: Segment, prev_seg: Segment) -> str:
    if prev_seg.segment == seg.segment:
        return 'S'

    seg_sonority = seg.sonority
    prev_seg_sonority = prev_seg.sonority

    if prev_seg_sonority == seg_sonority:
        return '='
    elif prev_seg_sonority < seg_sonority:
        return '<'
    else: # prev_sonority > sonority_i
        return '>'


@cython.cfunc
@cython.returns(str)
@cython.locals(
    seg = object,
    next_seg = object,

)
def relative_post_sonority(seg: Segment, next_seg: Segment) -> str:
    if next_seg.segment == seg.segment:
        return 'S'

    next_seg_sonority = next_seg.sonority
    seg_sonority = seg.sonority
    if next_seg_sonority == seg_sonority:
        return '='
    elif next_seg_sonority < seg_sonority:
        return '>'
    else: # sonority_i > next_seg
        return '<'


@cython.cfunc
@cython.returns(str)
def relative_sonority(seg: Segment,
                      prev_seg: Segment | None = None,
                      next_seg: Segment | None = None,
                      ) -> str:
    assert prev_seg is not None or next_seg is not None
    if prev_seg is None:
        prev_son = BOUNDARY_TOKEN
    else:
        prev_son = relative_prev_sonority(seg, prev_seg)
    if next_seg is None:
        post_son = BOUNDARY_TOKEN
    else:
        post_son = relative_post_sonority(seg, next_seg)
    return f'{prev_son}{BASE_SEGMENT_ENV}{post_son}'

# PHONOLOGICAL ENVIRONMENT
@cython.final
@cython.cclass
class PhonEnv:
    gap_ch: str | None = cython.declare(str)
    index: cython.Py_ssize_t = cython.declare(cython.Py_ssize_t)
    segment_i: Segment | None = cython.declare(object)
    supra_segs: list[Segment] = cython.declare(list)
    segments: list[Segment] = cython.declare(list)
    adjust_n: cython.Py_ssize_t = cython.declare(cython.Py_ssize_t)
    adjusted_index: int = cython.declare(cython.int)
    phon_env: str = cython.declare(str)

    @cython.locals(
        segments = list,
        i = cython.Py_ssize_t,
        gap_ch = Optional[str],
    )
    @cython.returns(Self)
    def __init__(self, segments, i, gap_ch=None):
        self.gap_ch = gap_ch
        if self.gap_ch:
            i, segments = self.preprocess_aligned_sequence(segments, i)
        self.index: cython.Py_ssize_t = i
        self.segment_i = None
        self.supra_segs = [_toSegment(s) if not self.is_gappy(s) else s for s in segments]
        self.segments, self.adjust_n = self.sep_segs_from_suprasegs(self.supra_segs, self.index)
        self.adjusted_index: cython.Py_ssize_t = self.index - self.adjust_n
        self.phon_env = self.get_phon_env()

    @cython.cfunc
    @cython.locals(
        segments = list,
        i = cython.Py_ssize_t,
        minus_offset = cython.Py_ssize_t,
        plus_offset = cython.Py_ssize_t,
        adj_segments = list,
        segment = object
    )
    @cython.returns(tuple[cython.Py_ssize_t, list])
    def preprocess_aligned_sequence(self, segments: list,
                                    i: cython.Py_ssize_t) -> tuple[cython.Py_ssize_t, list]:
        """Drop gaps and boundaries and flatten complex ngrams."""
        minus_offset, plus_offset = 0, 0
        adj_segments = []
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
                    continue
                elif isinstance(segment, tuple):
                    adj_segments.extend(segment)
                else:
                    adj_segments.append(segment)
        return adjusted_i, adj_segments

    @cython.cfunc
    @cython.locals(
        segments = list,
        i = cython.Py_ssize_t,
        j = cython.Py_ssize_t,
        adjust_n = cython.Py_ssize_t,
        segs = list,
        seg  = object
    )
    @cython.returns(tuple[list, cython.Py_ssize_t])
    def sep_segs_from_suprasegs(self, segments, i) -> tuple[list, cython.Py_ssize_t]:
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

    @cython.cfunc
    @cython.locals(
        front = cython.bint,
        nasal = cython.bint,
        rhotic = cython.bint,
        accented = cython.bint,
        env = str,
        i = cython.Py_ssize_t,
        prev_segment = object,
        next_segment = object,
    )
    @cython.returns(str)
    def get_phon_env(self, front=True, nasal=True, rhotic=True, accented=True) -> str:
        """Returns a string representing the phonological environment of a segment within a word"""
        
        # Tonemes/suprasegmentals
        if isinstance(self.segment_i, Segment) and self.segment_i.phone_class in ('TONEME', 'SUPRASEGMENTAL'):
            return BASE_TONEME_ENV
        
        # Word-initial segments (free-standing segments also considered word-initial)
        env: str = BASE_SEGMENT_ENV
        i: cython.Py_ssize_t = self.adjusted_index
        if i == 0:
            if len(self.segments) > 1:
                next_segment = self.segments[i+1]
            else:
                next_segment = None

            # Word-initial segments: #S
            if next_segment:
                if not self.is_gappy(self.segments[self.adjusted_index]):
                    env = self.relative_sonority(prev_seg=None,
                                                 next_seg=next_segment)
            
                # Add front vowel environment
                if front:
                    env = self.add_front_env(env, next_segment.base, suffix=True, prefix=None)
                # Add following nasal environment
                if nasal:
                    env = self.add_nasal_env(env, next_segment.base, suffix=True)
                # Add following rhotic environment
                if rhotic:
                    env = self.add_rhotic_env(env, next_segment.base, suffix=True)
                # Add accented/prosodically marked environment
                if accented:
                    env = self.add_accented_env(env, self.supra_segs[self.index + 1], suffix=True)
                    
                # # Add the next segment itself
                # env += '_' + next_segment.segment
                        
                return env
            
            # Free-standing segments
            else:
                return f'{BOUNDARY_TOKEN}{BASE_SEGMENT_ENV}{BOUNDARY_TOKEN}'
        
        # Word-final segments: S#
        elif i == len(self.segments)-1:
            assert len(self.segments) > 1
            prev_segment: Segment = self.segments[i-1]
            if not self.is_gappy(self.segments[self.adjusted_index]):
                env = self.relative_sonority(prev_seg=prev_segment)

            # Add front vowel environment
            if front:
                env = self.add_front_env(env, prev_segment.base, prefix=True)
                            
            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env
        
        # Word-medial segments
        else:
            prev_segment, next_segment = self.segments[i-1], self.segments[i+1]
            if not self.is_gappy(self.segments[self.adjusted_index]):
                env = self.relative_sonority(prev_seg=prev_segment, next_seg=next_segment)

            # Add front vowel environment
            if front:
                env = self.add_front_env(env, prev_segment.base, prefix=True, suffix=None)
                env = self.add_front_env(env, next_segment.base, suffix=True, prefix=None)
            # Add following nasal environment
            if nasal:
                env = self.add_nasal_env(env, next_segment.base, suffix=True)
            # Add following rhotic environment
            if rhotic:
                env = self.add_rhotic_env(env, next_segment.base, suffix=True)

            # Add accented/prosodically marked environment
            if accented:
                env = self.add_accented_env(env, self.supra_segs[self.index + 1], suffix=True)
                
            # # Add the next segment itself
            # env += '_' + next_segment.segment
                    
            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env

    @cython.cfunc
    def is_gappy(self, seg):
        return isinstance(seg, str) and (seg == self.gap_ch or BOUNDARY_TOKEN in seg)

    @cython.cfunc
    @cython.locals(
        prev_seg = Optional[object],
        next_seg = Optional[object],
    )
    def relative_sonority(self, prev_seg=None, next_seg=None):
        return relative_sonority(self.segment_i, prev_seg=prev_seg, next_seg=next_seg)

    @cython.cfunc
    def relative_prev_sonority(self, prev_seg):
        return relative_prev_sonority(self.segment_i, prev_seg)

    @cython.cfunc
    def relative_post_sonority(self, next_seg):
        return relative_post_sonority(self.segment_i, next_seg)

    @cython.cfunc
    @cython.locals(
        env = str,
        ch = object,
        symbol = str,
        regex = Optional[object],
        ch_list = Optional[set],
        phone_class = Optional[tuple]
    )
    @cython.returns(str)
    def add_env(self,
                env: str,
                ch: Segment,
                symbol: str,
                regex: Optional[re.Pattern[str]] = None,
                ch_list: Optional[set] = None,
                phone_class: Optional[tuple] = None,
                prefix=None,
                suffix=None,
                sep='_') -> str:
        assert prefix is not None or suffix is not None
        if phone_class and ch.phone_class in phone_class:
            env_match = True
        elif _is_env(ch=ch, regex=regex, ch_list=ch_list):
            env_match = True
        else:
            return env
        if prefix:
            return symbol + sep + env
        else: # suffix
            return env + sep + symbol

    @cython.cfunc
    @cython.returns(str)
    def add_front_env(self, env, ch, prefix: Optional[bool] = None,
                      suffix: Optional[bool] = None,
                      phone_class: Optional[set] = None) -> str:
        return self.add_env(env, ch, symbol='F', regex=front_vowel_regex, ch_list={'j', 'ɥ'},
                            prefix=prefix, suffix=suffix, phone_class=phone_class)

    @cython.cfunc
    @cython.returns(str)
    def add_nasal_env(self, env, ch, suffix: bool) -> str:
        return self.add_env(env, ch, symbol='N', regex=nasal_regex, suffix=suffix,
                            ch_list=None, phone_class=None, prefix=None)

    @cython.cfunc
    @cython.returns(str)
    def add_rhotic_env(self, env, ch, suffix: bool) -> str:
        return self.add_env(env, ch, symbol='R', regex=rhotic_regex, suffix=suffix,
                            ch_list=None, phone_class=None, prefix=None)

    @cython.cfunc
    @cython.locals(
        env = str,
        seg = object,
        suffix = cython.bint,
    )
    @cython.returns(str)
    def add_accented_env(self, env: str,
                         seg: Segment,
                         suffix: bool) -> str:
        return self.add_env(env, seg,
                            symbol='A',
                            phone_class=('TONEME', 'SUPRASEGMENTAL'), suffix=suffix,
                            prefix=None, regex=None, ch_list=None)
        
    def ngrams(self, exclude=set()):
        """Returns set of phonological environment strings of equal and lower order, 
        e.g. ">|S|#" -> ">|S", "S|#", ">|S|#"

        Returns:
            set: possible equal and lower order phonological environment strings
        """
        return phon_env_ngrams(self.phon_env, exclude=exclude)
    
    def __str__(self):
        return self.phon_env


@cython.ccall
@cython.locals(value = str, parts = list,
               i = cython.Py_ssize_t,
               x = tuple[str, cython.Py_ssize_t])
@cython.returns(set)
def join_n_fix(value) -> set:
    parts: list = value.split('_')
    result: set = set()
    for i in range(1, len(parts) + 1):
        for x in combinations(parts, i):
            result.add('_'.join(x))
    result.add('')
    return result


@cython.ccall
@cython.locals(
    phonEnv = str, exclude = set,
    prefix = str, base = str, suffix = str,
    prefixes = set, suffixes = set, ngrams = set,
)
@cython.returns(list)
def phon_env_ngrams(phonEnv: str, exclude: set) -> list:
    """Returns list of phonological environment strings of equal and lower order,
    e.g. ">|S|#" -> ">|S", "S|#", ">|S|#"

    Returns:
        list of possible equal and lower order phonological environment strings
    """
    if re.search(r'.+\|S\|.*', phonEnv) or re.search(r'.*\|S\|.+', phonEnv):
        prefix, base, suffix = phonEnv.split('|')
        prefixes = join_n_fix(prefix)
        suffixes = join_n_fix(suffix)
        ngrams = set()
        for prefix in prefixes:
            for suffix in suffixes:
                ngrams.add(f'{prefix}|S|{suffix}')
    else:
        assert phonEnv in (BASE_TONEME_ENV, BASE_SEGMENT_ENV)
        ngrams: set = {phonEnv}
    
    if len(exclude) > 0:
        return [ngram for ngram in ngrams if ngram not in exclude]
    return list(ngrams)

@cython.ccall
@cython.returns(str)
def get_phon_env(segments, i) -> str:
    return PhonEnv(segments, i).phon_env
