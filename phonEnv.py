
import os
import sys

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load phonological constants initialized in initPhoneData.py
from PhoneticSimilarity.initPhoneData import tonemes, nasal_regex, front_vowel_regex
from PhoneticSimilarity.segment import _toSegment

# HELPER FUNCTIONS
def _is_front_env(ch):
    if front_vowel_regex.search(ch) or ch in {'j', 'ɥ'}:
        return True
    return False


def _is_nasal_env(ch):
    if nasal_regex.search(ch):
        return True
    return False


# PHONOLOGICAL ENVIRONMENT
def get_phon_env(segments, i):
    """Returns a string representing the phonological environment of a segment within a word"""
    # Convert IPA strings to Segment objects and get base segment
    segments = [_toSegment(seg) for seg in segments]
    segment_i = segments[i]
    base = segment_i.base

    # Tonemes
    if base in tonemes: 
        # TODO should remove all tonemes from word and reevaluate without them, so that final segments are considered final despite being "followed" by a toneme
        return '|T|'

    # Word-initial segments (free-standing segments also considered word-initial)
    elif i == 0:
        if len(segments) > 1:
            next_segment = segments[i+1]
            sonority_i, next_sonority = segment_i.sonority, next_segment.sonority
        else:
            next_segment = None

        # Word-initial segments: #S
        if next_segment:
            if segment_i == next_segment:
                env = '#|S|S'
            elif sonority_i == next_sonority:
                env = '#|S|='
            elif sonority_i > next_sonority:
                env = '#|S|>'
            else: # sonority_i < next_sonority:
                env = '#|S|<'
        
            # Add front vowel environment
            if _is_front_env(next_segment.base):
                env += '_F'
            # Add following nasal environment
            if _is_nasal_env(next_segment.base):
                env += '_N'
                
            # # Add the next segment itself
            # env += '_' + next_segment.segment
                    
            return env
        
        # Free-standing segments
        else:
            return '#|S|#'
    
    # Word-final segments: S#
    elif i == len(segments)-1:
        assert len(segments) > 1
        prev_segment = segments[i-1]
        if prev_segment == segment_i:
            env = 'S|S|#'
        else:
            prev_sonority, sonority_i = prev_segment.sonority, segment_i.sonority
            
            if prev_sonority == sonority_i:
                env = '=|S|#'

            elif prev_sonority < sonority_i:
                env = '<|S|#'

            else: # prev_sonority > sonority_i
                env = '>|S|#' 

        # Add front vowel environment
        if _is_front_env(prev_segment.base):
            env = 'F_' + env
                        
        # # Add the previous segment itself
        # env = prev_segment.segment + '_' + env
        
        return env
    
    # Word-medial segments
    else:
        prev_segment, next_segment = segments[i-1], segments[i+1]
        prev_sonority, sonority_i, next_sonority = prev_segment.sonority, segment_i.sonority, next_segment.sonority
        
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
            if segment_i == next_segment:
                env = '<|S|S'
            else:
                env = '<|S|='
        
        elif prev_sonority > sonority_i == next_sonority:
            if segment_i == next_segment:
                env = '>|S|S'
            else:
                env = '>|S|='
        
        elif prev_sonority == sonority_i < next_sonority:
            if segment_i == prev_segment:
                env = 'S|S|<'
            else:
                env = '=|S|<'
        
        elif prev_sonority == sonority_i > next_sonority:
            if segment_i == prev_segment:
                env = 'S|S|>'
            else:
                env = '=|S|>'
        
        elif prev_sonority == sonority_i == next_sonority:
            if segment_i == prev_segment:
                env = 'S|S|='
            elif segment_i == next_segment:
                env = '=|S|S'
            else:
                env = '=|S|='
        
        else:
            raise ValueError(f'Unable to determine environment for segment {i} /{segments[i].segment}/ within /{"".join([seg.segment for seg in segments])}/')
        
        # Add front vowel environment
        if _is_front_env(prev_segment.base):
            env = 'F_' + env
        if _is_front_env(next_segment.base):
            env += '_F'
        # Add following nasal environment
        if _is_nasal_env(next_segment.base):
            env += '_N'
            
        # # Add the next segment itself
        # env += '_' + next_segment.segment
                
        # # Add the previous segment itself
        # env = prev_segment.segment + '_' + env
        
        return env