
import os
import sys

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load phonological constants initialized in initPhoneData.py
from phonUtils.initPhoneData import tonemes, nasal_regex, rhotic_regex, front_vowel_regex, diacritics
from phonUtils.segment import _toSegment, _is_vowel
from phonUtils.syllables import syllabify

# HELPER FUNCTIONS
def _is_env(ch, regex=None, ch_list=None):
    if regex and regex.search(ch):
        return True
    if ch_list and ch in ch_list:
        return True
    return False

# Relative sonority functions
def relative_prev_sonority(seg, prev_seg):
    if prev_seg == seg:
        return 'S'
    elif prev_seg.sonority == seg:
        return '='
    elif prev_seg.sonority < seg:
        return '<'
    else: # prev_sonority > sonority_i
        return '>'
    
def relative_post_sonority(seg, next_seg):
    if next_seg == seg:
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
        prev_son = '#'
    else:
        prev_son = relative_prev_sonority(seg, prev_seg)
    if next_seg is None:
        post_son = '#'
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
    
    def get_phon_env(self, front=True, nasal=True, rhotic=True, accented=True):
        """Returns a string representing the phonological environment of a segment within a word"""
        
        # Tonemes/suprasegmentals
        if self.segment_i.phone_class in ('TONEME', 'SUPRASEGMENTAL'):
            return '|T|'
        
        # Word-initial segments (free-standing segments also considered word-initial)
        i = self.adjusted_index
        if i == 0:
            if len(self.segments) > 1:
                next_segment = self.segments[i+1]
            else:
                next_segment = None

            # Word-initial segments: #S
            if next_segment:
                env = self.relative_sonority(next_seg=next_segment)
            
                # Add front vowel environment
                if front:
                    env = self.add_front_env(env, next_segment.base, suffix=True)
                # Add following nasal environment
                if nasal:
                    env = self.add_nasal_env(env, next_segment.base, suffix=True)
                # Add following rhotic environment
                if rhotic:
                    env = self.add_rhotic_env(env, next_segment.base, suffix=True)
                    
                # # Add the next segment itself
                # env += '_' + next_segment.segment
                        
                return env
            
            # Free-standing segments
            else:
                return '#|S|#'
        
        # Word-final segments: S#
        elif i == len(self.segments)-1:
            assert len(self.segments) > 1
            prev_segment = self.segments[i-1]
            env = self.relative_sonority(prev_seg=prev_segment)

            # Add front vowel environment
            if front:
                env = self.add_front_env(env, prev_segment.base, prefix=True)
            
            # Add accented/prosodically marked environment
            if accented:
                env = self.add_accented_env(env, self.supra_segs[self.index-1], prefix=True)
                            
            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env
        
        # Word-medial segments
        else:
            prev_segment, next_segment = self.segments[i-1], self.segments[i+1]
            env = self.relative_sonority(prev_seg=prev_segment, next_seg=next_segment)

            # Add front vowel environment
            if front:
                env = self.add_front_env(env, prev_segment.base, prefix=True)
                env = self.add_front_env(env, next_segment.base, suffix=True)
            # Add following nasal environment
            if nasal:
                env = self.add_nasal_env(env, next_segment.base, suffix=True)
            # Add following rhotic environment
            if rhotic:
                env = self.add_rhotic_env(env, next_segment.base, suffix=True)

            # Add accented/prosodically marked environment
            if accented:
                env = self.add_accented_env(env, self.supra_segs[self.index-1], prefix=True)
                
            # # Add the next segment itself
            # env += '_' + next_segment.segment
                    
            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env
    
    def relative_sonority(self, prev_seg=None, next_seg=None):
        return relative_sonority(self.sefment_i, prev_seg=prev_seg, next_seg=next_seg)
    
    def relative_prev_sonority(self, prev_seg):
        return relative_prev_sonority(self.segment_i, prev_seg)

    def relative_post_sonority(self, next_seg):
        return relative_post_sonority(self.segment_i, next_seg)

    def add_env(self, 
                env, ch, symbol, 
                regex=None, 
                ch_list=None, 
                phone_class=None, 
                prefix=None, suffix=None, 
                sep='_'):
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
    
    def add_front_env(self, env, ch, **kwargs):
        return self.add_env(env, ch, symbol='F', regex=front_vowel_regex, ch_list={'j', 'ɥ'}, **kwargs)
    
    def add_nasal_env(self, env, ch, **kwargs):
        return self.add_env(env, ch, symbol='N', regex=nasal_regex, **kwargs)
    
    def add_rhotic_env(self, env, ch, **kwargs):
        return self.add_env(env, ch, symbol='R', regex=rhotic_regex, **kwargs)
    
    def add_accented_env(self, env, seg, **kwargs):
        return self.add_env(env, seg, symbol='A', phone_class=('TONEME', 'SUPRASEGMENTAL'), **kwargs)
        
    def __str__(self):
        return self.phon_env
        

def get_phon_env(segments, i, **kwargs):
    phon_env = PhonEnv(segments, i, **kwargs)
    return phon_env.phon_env