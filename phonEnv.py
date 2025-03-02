import os
import re
import sys
from itertools import combinations

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load phonological constants initialized in initPhoneData.py
from phonUtils.initPhoneData import (alveolar, alveolopalatal, bilabial,
                                     dental, epiglottal, front_vowel_regex,
                                     glottal, labiodental, lateral, liquids,
                                     nasal_regex, palatal, pharyngeal,
                                     postalveolar, retroflex, rhotic_regex,
                                     uvular, velar)
from phonUtils.segment import Segment, _toSegment
from phonUtils.syllables import syllabify

# CONSTANTS
PHON_ENV_REGEX = re.compile(r'.*\|[ST]\|.*')
BASE_SEGMENT_ENV = '|S|'
BASE_TONEME_ENV = '|T|'
BOUNDARY_TOKEN = "#"
PHON_ENV_MAP = {
    "FRONT": {
        "symbol": "F",
        "regex": front_vowel_regex,
        "ch_list": {'j', 'ɥ'},
    },
    "NASAL": {
        "symbol": "N",
        "regex": nasal_regex,
    },
    # "RHOTIC": {
    #     "symbol": "R",
    #     "regex": rhotic_regex,
    # },
    # "LATERAL": {
    #     "symbol": "L",
    #     "regex": re.compile(f"[{''.join(lateral)}ˡ]"),
    # },
    "LIQUID": {
        "symbol": "RL",
        "regex": re.compile(f"[{''.join(liquids)}]"),
    },
    "LABIAL": {
        "symbol": "B",
        "regex": re.compile(f"[{''.join(bilabial.union(labiodental))}ʷᵝ]"),
    },
    # "DENTAL/ALVEOLAR": {
    #     "symbol": "D",
    #     "regex": re.compile(f"[{''.join(dental.union(alveolar))}]"),
    # },
    # "POST-ALVEOLAR": {
    #     "symbol": "Š",
    #     "regex": re.compile(f"[{''.join(postalveolar.union(retroflex))}]"),
    # },
    # "PALATAL": {
    #     "symbol": "P",
    #     "regex": re.compile(f"[{''.join(palatal.union(alveolopalatal))}]"),
    # },
    # "VELAR/UVULAR": {
    #     "symbol": "K",
    #     "regex": re.compile(f"[{''.join(velar.union(uvular))}ˠ]"),
    # },
    # "PHARYNGEAL/(EPI)GLOTTAL": {
    #     "symbol": "H",
    #     "regex": re.compile(f"[{''.join(glottal.union(epiglottal).union(pharyngeal))}ˤˀ]"),
    # },
    "VOICELESS": {
        "symbol": "-Voice",
        "features": {"periodicGlottalSource": 0},
    },
    # "VOICED": {
    #     "symbol": "+Voice",
    #     "features": {"periodicGlottalSource": 1},
    # },
    "VOWEL": {
        "symbol": "V",
        "phone_class": ['VOWEL', 'DIPHTHONG'],
    },
    "CONSONANT": {
        "symbol": "C",
        "phone_class": ['CONSONANT', 'GLIDE'],
    },
    # "ACCENTED": {
    #     "symbol": "A",
    #     "phone_class": ['TONEME', 'SUPRASEGMENTAL'],
    # },
}

# syllable onset: SylO
# syllable medial: SylM
# syllable coda: SylC


# HELPER FUNCTIONS
def _is_env(segment: Segment, regex=None, ch_list=None, features=None):
    if features:
        if all(segment.features.get(feature) == feature_val for feature, feature_val in features.items()):
            return True
        return False
    if regex and regex.search(segment.segment):
        return True
    if ch_list and segment.base in ch_list:
        return True
    return False

# Relative sonority functions
def relative_prev_sonority(seg, prev_seg):
    if prev_seg == seg.sonority:
        return 'S'
    elif prev_seg.sonority == seg.sonority:
        return '='
    elif prev_seg.sonority < seg.sonority:
        return '<'
    else: # prev_sonority > sonority_i
        return '>'
    
def relative_post_sonority(seg, next_seg):
    if next_seg == seg.sonority:
            return 'S'
    elif next_seg.sonority == seg.sonority:
        return '='
    elif next_seg.sonority < seg.sonority:
        return '>'
    else: # sonority_i > next_seg
        return '<'

def relative_sonority(seg, prev_seg=None, next_seg=None):
    assert prev_seg is not None or next_seg is not None
    if prev_seg is None:
        prev_son = BOUNDARY_TOKEN
    else:
        prev_son = relative_prev_sonority(seg, prev_seg)
    if next_seg is None:
        post_son = BOUNDARY_TOKEN
    else:
        post_son = relative_post_sonority(seg, next_seg)
    return f'{prev_son}|S|{post_son}'

# PHONOLOGICAL ENVIRONMENT
class PhonEnv:
    def __init__(self, segments, i, **kwargs):
        self.index = i
        self.segment_i = None
        self.supra_segs = [_toSegment(s) for s in segments]
        self.segments, self.adjust_n = self.sep_segs_from_suprasegs(self.supra_segs, self.index)
        self.adjusted_index = self.index - self.adjust_n
        self.phon_env = self.get_phon_env(**kwargs)
    
    def sep_segs_from_suprasegs(self, segments, i):
        adjust_n = 0
        segs = []
        for j, seg in enumerate(segments):
            if seg.phone_class not in ('TONEME', 'SUPRASEGMENTAL'):
                segs.append(seg)
            elif j < i:
                adjust_n += 1
            if j == i:
                self.segment_i = seg
        return segs, adjust_n
    
    def get_phon_env(self):
        """Returns a string representing the phonological environment of a segment within a word"""
        
        # Tonemes/suprasegmentals
        if self.segment_i.phone_class in ('TONEME', 'SUPRASEGMENTAL'):
            return BASE_TONEME_ENV
        else:
            env = BASE_SEGMENT_ENV
        
        # Word-initial segments (free-standing segments also considered word-initial)
        i = self.adjusted_index
        if i == 0:
            if len(self.segments) > 1:
                next_segment = self.segments[i+1]
            else:
                next_segment = None

            # Word-initial segments: #S
            if next_segment:
                #env = self.relative_sonority(next_seg=next_segment)
                env = BOUNDARY_TOKEN + env
            
                # Add feature environments
                env = self.add_envs(env, next_segment, suffix=True)
                    
                # # Add the next segment itself
                # env += '_' + next_segment.segment
                        
                return env
            
            # Free-standing segments
            else:
                return f"{BOUNDARY_TOKEN}{BASE_SEGMENT_ENV}{BOUNDARY_TOKEN}"
        
        # Word-final segments: S#
        elif i == len(self.segments)-1:
            assert len(self.segments) > 1
            prev_segment = self.segments[i-1]
            #env = self.relative_sonority(prev_seg=prev_segment)
            env += BOUNDARY_TOKEN

            # Add feature environments
            env = self.add_envs(env, prev_segment, prefix=True)

            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env
        
        # Word-medial segments
        else:
            prev_segment, next_segment = self.segments[i-1], self.segments[i+1]
            #env = self.relative_sonority(prev_seg=prev_segment, next_seg=next_segment)

            # Add feature environments
            env = self.add_envs(env, prev_segment, prefix=True)
            env = self.add_envs(env, next_segment, suffix=True)

            # # Add the next segment itself
            # env += '_' + next_segment.segment
                    
            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env
    
    def relative_sonority(self, prev_seg=None, next_seg=None):
        return relative_sonority(self.segment_i, prev_seg=prev_seg, next_seg=next_seg)
    
    def relative_prev_sonority(self, prev_seg):
        return relative_prev_sonority(self.segment_i, prev_seg)

    def relative_post_sonority(self, next_seg):
        return relative_post_sonority(self.segment_i, next_seg)

    def add_env(self, 
                env,
                segment,
                symbol, 
                regex=None, 
                ch_list=None, 
                phone_class=None, 
                features=None,
                prefix=None,
                suffix=None,
                sep='_'):
        assert prefix is not None or suffix is not None
        if phone_class and segment.phone_class in phone_class:
            env_match = True
        elif _is_env(segment=segment, features=features, regex=regex, ch_list=ch_list):
            env_match = True
        else:
            return env
        if prefix:
            return symbol + sep + env
        else: # suffix
            return env + sep + symbol
    
    def add_envs(self, env, segment, **kwargs):
        for _, encoding_map in PHON_ENV_MAP.items():
            symbol = encoding_map["symbol"]
            regex = encoding_map.get("regex", None)
            ch_list = encoding_map.get("ch_list", None)
            phone_class = encoding_map.get("phone_class", None)
            features = encoding_map.get("features", None)
            env = self.add_env(
                env,
                segment,
                symbol=symbol,
                regex=regex,
                ch_list=ch_list,
                phone_class=phone_class,
                features=features,
                **kwargs
            )
        return env
    
    def add_front_env(self, env, ch, **kwargs):
        return self.add_env(env, ch, symbol='F', regex=front_vowel_regex, ch_list={'j', 'ɥ'}, **kwargs)
    
    def add_nasal_env(self, env, ch, **kwargs):
        return self.add_env(env, ch, symbol='N', regex=nasal_regex, **kwargs)
    
    def add_rhotic_env(self, env, ch, **kwargs):
        return self.add_env(env, ch, symbol='R', regex=rhotic_regex, **kwargs)
    
    def add_accented_env(self, env, seg, **kwargs):
        return self.add_env(env, seg, symbol='A', phone_class=('TONEME', 'SUPRASEGMENTAL'), **kwargs)
        
    def ngrams(self, exclude=set()):
        """Returns set of phonological environment strings of equal and lower order, 
        e.g. ">|S|#" -> ">|S", "S|#", ">|S|#"

        Returns:
            set: possible equal and lower order phonological environment strings
        """
        return phon_env_ngrams(self.phon_env, exclude=exclude)
    
    def __str__(self):
        return self.phon_env
        
def phon_env_ngrams(phonEnv, exclude=set()):
    """Returns set of phonological environment strings of equal and lower order, 
    e.g. ">|S|#" -> ">|S", "S|#", ">|S|#"

    Returns:
        set: possible equal and lower order phonological environment strings
    """
    if re.search(r'.*\|S\|.*', phonEnv):
        prefix, base, suffix = phonEnv.split('|')
        prefix = prefix.split('_')
        prefixes = set()
        for i in range(1, len(prefix)+1):
            for x in combinations(prefix, i):
                prefixes.add('_'.join(x))
        prefixes.add('')
        suffix = suffix.split('_')
        suffixes = set()
        for i in range(1, len(suffix)+1):
            for x in combinations(suffix, i):
                suffixes.add('_'.join(x))
        suffixes.add('')
        ngrams = set()
        for prefix in prefixes:
            for suffix in suffixes:
                ngrams.add(f'{prefix}|S|{suffix}')
    else:
        assert phonEnv in (BASE_TONEME_ENV, BASE_SEGMENT_ENV)
        ngrams = [phonEnv]
    
    if len(exclude) > 0:
        return [ngram for ngram in ngrams if ngram not in exclude]
    return ngrams

def get_phon_env(segments, i, **kwargs):
    phon_env = PhonEnv(segments, i, **kwargs)
    return phon_env.phon_env
