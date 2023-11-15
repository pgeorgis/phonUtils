
import os
import sys

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load phonological constants initialized in initPhoneData.py
from phonUtils.initPhoneData import tonemes, nasal_regex, rhotic_regex, front_vowel_regex, diacritics
from phonUtils.segment import _toSegment, _is_vowel
from phonUtils.syllables import syllabify

# HELPER FUNCTIONS
def _is_front_env(ch):
    if front_vowel_regex.search(ch) or ch in {'j', 'ɥ'}:
        return True
    return False

def _is_nasal_env(ch):
    if nasal_regex.search(ch):
        return True
    return False

def _is_rhotic_env(ch):
    if rhotic_regex.search(ch):
        return True
    return False


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
                sonority_i, next_sonority = self.segment_i.sonority, next_segment.sonority
            else:
                next_segment = None

            # Word-initial segments: #S
            if next_segment:
                if self.segment_i == next_segment:
                    env = '#|S|S'
                elif sonority_i == next_sonority:
                    env = '#|S|='
                elif sonority_i > next_sonority:
                    env = '#|S|>'
                else: # sonority_i < next_sonority:
                    env = '#|S|<'
            
                # Add front vowel environment
                if front and _is_front_env(next_segment.base):
                    env += '_F'
                # Add following nasal environment
                if nasal and _is_nasal_env(next_segment.base):
                    env += '_N'
                # Add following rhotic environment
                if rhotic and _is_rhotic_env(next_segment.base):
                    env += '_R'
                    
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
            if prev_segment == self.segment_i:
                env = 'S|S|#'
            else:
                prev_sonority, sonority_i = prev_segment.sonority, self.segment_i.sonority
                
                if prev_sonority == sonority_i:
                    env = '=|S|#'

                elif prev_sonority < sonority_i:
                    env = '<|S|#'

                else: # prev_sonority > sonority_i
                    env = '>|S|#' 

            # Add front vowel environment
            if front and _is_front_env(prev_segment.base):
                env = 'F_' + env
            
            # Add accented/prosodically marked environment
            if accented and self.supra_segs[self.index-1].phone_class in ('TONEME', 'SUPRASEGMENTAL'):
                env = 'A_' + env
                            
            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env
        
        # Word-medial segments
        else:
            prev_segment, next_segment = self.segments[i-1], self.segments[i+1]
            prev_sonority, sonority_i, next_sonority = prev_segment.sonority, self.segment_i.sonority, next_segment.sonority
            
            # Sonority plateau: =S=
            if prev_segment == sonority_i == next_sonority:
                env = '=|S|='
            
            # Sonority peak: <S>
            elif prev_sonority < sonority_i > next_sonority:
                env = '<|S|>'
            
            # Sonority trench: >S< # TODO is this the best term?
            elif prev_sonority > sonority_i < next_sonority:
                env = '>|S|<'
            
            # Descending sonority: >S>
            elif prev_sonority > sonority_i > next_sonority:
                env = '>|S|>'
            
            # Ascending sonority: <S<
            elif prev_sonority < sonority_i < next_sonority:
                env = '<|S|<'
            
            elif prev_sonority < sonority_i == next_sonority:
                if self.segment_i == next_segment:
                    env = '<|S|S'
                else:
                    env = '<|S|='
            
            elif prev_sonority > sonority_i == next_sonority:
                if self.segment_i == next_segment:
                    env = '>|S|S'
                else:
                    env = '>|S|='
            
            elif prev_sonority == sonority_i < next_sonority:
                if self.segment_i == prev_segment:
                    env = 'S|S|<'
                else:
                    env = '=|S|<'
            
            elif prev_sonority == sonority_i > next_sonority:
                if self.segment_i == prev_segment:
                    env = 'S|S|>'
                else:
                    env = '=|S|>'
            
            elif prev_sonority == sonority_i == next_sonority:
                if self.segment_i == prev_segment:
                    env = 'S|S|='
                elif self.segment_i == next_segment:
                    env = '=|S|S'
                else:
                    env = '=|S|='
            
            else:
                raise ValueError(f'Unable to determine environment for segment {i} /{self.segments[i].segment}/ within /{"".join([seg.segment for seg in self.segments])}/')
            
            # Add front vowel environment
            if front and _is_front_env(prev_segment.base):
                env = 'F_' + env
            if front and _is_front_env(next_segment.base):
                env += '_F'
            # Add following nasal environment
            if nasal and _is_nasal_env(next_segment.base):
                env += '_N'
            # Add following rhotic environment
            if rhotic and _is_rhotic_env(next_segment.base):
                env += '_R'

            # Add accented/prosodically marked environment
            if accented and self.supra_segs[self.index-1].phone_class in ('TONEME', 'SUPRASEGMENTAL'):
                env = 'A_' + env
                
            # # Add the next segment itself
            # env += '_' + next_segment.segment
                    
            # # Add the previous segment itself
            # env = prev_segment.segment + '_' + env
            
            return env
        
    def __str__(self):
        return self.phon_env
        

def get_phon_env(segments, i, **kwargs):
    phon_env = PhonEnv(segments, i, **kwargs)
    return phon_env.phon_env