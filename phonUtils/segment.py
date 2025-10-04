import re
import os
import threading
from collections import defaultdict
from functools import lru_cache
from typing import Iterable

from .constants import (
    # Top-level phone sets
    VOWELS, GLIDES, CONSONANTS, TONEMES,
    # Phone classes by manner of articulation
    PLOSIVES, IMPLOSIVES, NASALS, AFFRICATES, FRICATIVES, TRILLS, TAPS_AND_FLAPS, APPROXIMANTS, CLICKS,
    # Phone classes by place of articulation
    BILABIALS, ALVEOLARS, LATERALS, POSTALVEOLARS, ALVEOLOPALATALS, RETROFLEXES, PALATALS, VELARS, UVULARS, PHARYNGEALS, EPIGLOTTALS, GLOTTALS,
    # Diacritics and associated features
    DIACRITICS_EFFECTS,
    # Phonological features and feature geometry weights 
    PHONE_FEATURES, TONE_DIACRITICS_MAP, TONE_LEVELS,
    # IPA regexes and constants
    IPA_SEGMENTS, SEGMENT_REGEX,
    PREASPIRATION_REGEX, DIPHTHONG_REGEX, TONEME_REGEX, AFFRICATE_REGEX,
    FRONT_VOWEL_REGEX, CENTRAL_VOWEL_REGEX, BACK_VOWEL_REGEX,
    CLOSE_VOWEL_REGEX, CLOSE_MID_VOWEL_REGEX, OPEN_VOWEL_REGEX, OPEN_MID_VOWEL_REGEX,
    # Phone features
    FEATURE_SET,
)
from .ipaTools import strip_diacritics, normalize_ipa_ch, verify_charset


def segment_in_group(ipa_str: str, group: Iterable):
    """Checks if a base IPA character is a member of a list or set."""
    assert isinstance(ipa_str, str)
    if strip_diacritics(ipa_str)[0] in group:
        return True
    else:
        return False


def segment_is_vowel(segment: str) -> bool:
    """Returns True if segment is a vowel."""
    return segment_in_group(segment, VOWELS)


def segment_is_affricate(phone: str) -> bool:
    if '͡' in phone:
        if AFFRICATE_REGEX.search(phone):
            return True
    elif phone in AFFRICATES:
        return True
    return False


class Segment:
    _cache = {}
    _lock = threading.Lock()

    def __new__(cls, segment, normalize=False):
        # Optionally normalize IPA string input and check for non-IPA characters
        if normalize:
            segment = cls.normalize(segment)

        # Check for existing cached Segment instance from input
        with cls._lock:
            if segment in cls._cache:
                return cls._cache[segment]

            # Create a new Segment instance if not cached
            instance = super().__new__(cls)
            instance.normalized = segment

        return instance

    def __init__(self, segment):
        # Prevent re-initializing cached objects
        if getattr(self, "_initialized", False) is True:
            return

        # Get optionally normalized IPA string
        self.segment = getattr(self, "normalized", segment)

        # Base segment: no diacritics; first element of diphthongs, affricates, or complex consonants
        self.stripped, self.base = self.get_base_ch()

        # Get distinctive phonological feature dictionary
        self.features = self.get_phone_features(self.segment)
        
        # Get phone class
        self.phone_class = self.get_phone_class()

        # Get voicing status, manner, and place of articulation
        if self.phone_class not in ('TONEME', 'SUPRASEGMENTAL'):
            self.voiced = self.features['periodicGlottalSource'] == 1
            self.voiceless = self.features['periodicGlottalSource'] == 0
        else:
            self.voiced = None
            self.voiceless = None
        self.manner = self.get_manner()
        self.poa = self.get_poa()

        # Get sonority
        self.sonority = self.get_sonority()

        # Mark segment as initialized and add to cache
        self._initialized = True
        Segment._cache[self.segment] = self

    @staticmethod
    @lru_cache(maxsize=None)
    def normalize(segment: str) -> str:
        segment = normalize_ipa_ch(segment)
        verify_charset(segment)
        return segment

    def get_base_ch(self) -> tuple[str, str]:
        no_diacritics = strip_diacritics(self.segment)
        if len(no_diacritics) < 1:
            #raise ValueError(f'Error: invalid segment <{self.segment}>, no base IPA character found!')
            return '', self.segment
        else:
            return no_diacritics, no_diacritics[0]

    def get_phone_class(self) -> str:
        if DIPHTHONG_REGEX.search(self.segment):
            return 'DIPHTHONG'
        elif self.base in GLIDES:
            return 'GLIDE'
        elif self.base in VOWELS:
            return 'VOWEL'
        elif self.base in CONSONANTS:
            return 'CONSONANT'
        elif self.base in TONEMES or self.base in TONE_DIACRITICS_MAP:
            return 'TONEME'
        elif all([ch in DIACRITICS_EFFECTS for ch in self.base]):
            return 'SUPRASEGMENTAL'
        else:
            raise ValueError(f'Could not determine phone class of {self.segment}')

    def get_phone_features(self, segment: str) -> dict:
        """Returns a dictionary of distinctive phonological feature values for the segment"""
        
        # Retrieve saved feature dictionary if already cached for this segment
        if segment in Segment._cache:
            return Segment._cache[segment].features

        # Generate an empty phone feature dictionary with default values of 0
        feature_dict = dict.fromkeys(PHONE_FEATURES[next(iter(PHONE_FEATURES))], 0)

        # Split segment into component parts, if relevant
        parts = segment.split('͡') if '͡' in segment else segment.split('͜')
        bases = []

        # Generate feature dictionary for each part and add to main feature dict
        for part in parts:
            part = part.strip()
            if part:
                # Base of the segment is the non-diacritic portion
                base = strip_diacritics(part)
                if not base:
                    #raise AssertionError(f'Error: invalid segment <{segment}>, no base IPA character found!')
                    return self.get_suprasegmental_features(segment)
                bases.append(base)

                # If the length of the base > 1, the segment is a diphthong (e.g. /e̯a/) or complex toneme (e.g. /˥˩/)
                # Filter out tonemes to handle diphthongs first
                if (len(base) > 1) and (base[0] not in TONEMES):
                    return self.get_diphthong_features(segment)

                # Handle tonemes
                elif base[0] in TONEMES:
                    return self.get_tonal_features(segment)

                # Otherwise, retrieve the base phone's features
                else:
                    base_id = PHONE_FEATURES[base]
                    modifiers = set(part) - set(base)
                    if modifiers:
                        part_id = self.apply_diacritics(base, base_id, modifiers)
                    else:
                        part_id = base_id

                    # Add to overall segment ID
                    for feature in part_id:
                        # Value = 1 (+) overrides value = 0 (-,0)
                        feature_dict[feature] = max(feature_dict[feature], part_id[feature])

        # Ensure that affricates are +DELAYED RELEASE and -CONTINUANT
        if len(parts) > 1:
            if bases[0] in PLOSIVES and bases[-1] in FRICATIVES:
                feature_dict['delayedRelease'] = 1
                feature_dict['continuant'] = 0

        return feature_dict

    def apply_diacritics(self, base: str, base_features: dict, diacritics: set) -> dict:
        """Applies feature values of diacritics to base segments

        Args:
            base (str): base IPA segment
            base_features (dict): feature dictionary of base segment
            diacritics (set): diacritics to apply
        """

        # Create a new dictionary to store the modified values
        modified_features = base_features.copy()

        # Apply diacritic effects to feature dictionary
        for modifier in diacritics:
            for feature, value in DIACRITICS_EFFECTS[modifier]:
                modified_features[feature] = value
                
            if modifier == '̞': # lowered diacritic: turns fricatives into approximants
                if base[0] in FRICATIVES:
                    modified_features['approximant'] = 1
                    modified_features['consonantal'] = 0
                    modified_features['delayedRelease'] = 0
                    modified_features['sonorant'] = 1
            
            elif modifier == '̝': # raised diacritic
                # turn approximants/trills into fricativized approximants
                if base[0] in APPROXIMANTS.union(TRILLS):
                    modified_features['delayedRelease'] = 1
                    
                # turn fricatives into plosives
                elif base[0] in FRICATIVES:
                    modified_features['continuant'] = 0
                    modified_features['delayedRelease'] = 0
        
        return modified_features

    def get_diphthong_features(self, diphthong: str) -> defaultdict:
        """Returns dictionary of features for diphthongal segment"""
        components = segment_ipa(diphthong, combine_diphthongs=False)

        # Create weights: 1 for syllabic components and 0.5 for non-syllabic components
        weights = [0.5 if '̯' in component else 1 for component in components]
        
        # Normalize the weights
        weight_sum = sum(weights)
        weights = [i/weight_sum for i in weights]
        
        # Create combined dictionary using features of component segments
        diphth_dict = defaultdict(lambda:0)
        for component, weight in zip(components, weights):
            feature_id = self.get_phone_features(component)
            for feature in feature_id:
                diphth_dict[feature] += (weight * feature_id[feature])
        
        # Length feature should be either 0 or 1
        if diphth_dict['long'] > 0:
            diphth_dict['long'] = 1
            
        return diphth_dict

    def get_tonal_features(self, toneme: str) -> defaultdict:
        """Computes complex tonal features"""
        
        # Set the base as the first component of the toneme
        base = toneme[0]
        
        # Create copy of original feature dictionary, or else it modifies the source
        toneme_id = defaultdict(lambda:0)
        for feature in PHONE_FEATURES[base]:
            toneme_id[feature] = PHONE_FEATURES[base][feature]
        
        # Get the tone level of each tonal component of the toneme
        toneme_levels = [TONE_LEVELS[t] for t in toneme if t in TONEMES]
        
        # Compute the complex tone features if not just a level tone
        if len(set(toneme_levels)) > 1:
            
            # Add feature tone_contour to all non-level tones
            toneme_id['tone_contour'] = 1
            
            # Ensure that contour tones do not have features tone_mid, which is unique to mid level tone
            # Note: Wang (1967) proposes that all contour tones also have tone_central=0, but if this is true
            # we cannot distinguish between, e.g. rising (˩˥), high rising (˦˥), and low rising (˩˨)
            toneme_id['tone_mid'] = 0
        
            # Get the maximum tonal level
            max_level = max(toneme_levels)
            
            # Add feature tone_high if the maximum tone level is at least 4
            if max_level >= 4:
                toneme_id['tone_high'] = 1
            
            # Add feature tone_central if any component tone is level 2-4
            if any(tone in {2,3,4} for tone in toneme_levels):
                toneme_id['tone_central'] = 1
            
            # Check whether any subsequence of the tonal components is rising or falling
            contours = {}
            for t in range(len(toneme_levels)-1):
                t_seq = toneme_levels[t:t+2]
                
                # Check for a tonal rise
                if t_seq[0] < t_seq[1]:
                    toneme_id['tone_rising'] = 1
                    contours[t] = 'rise'
                
                # Check for a tonal fall
                elif t_seq[0] > t_seq[1]:
                    toneme_id['tone_falling'] = 1
                    contours[t] = 'fall'
                    
                    # If a subsequence is falling, check whether the previous subsequence was rising
                    # in order to determine whether the tone is convex (rising-falling)
                    if t > 0:
                        if contours[t-1] == 'rise':
                            toneme_id['tone_convex'] = 1
                                                
                # Otherwise two equal tone levels in a row, e.g. '⁴⁴²'
                else:
                    contours[t] = 'level'
        
        return toneme_id

    def get_suprasegmental_features(self, supraseg: str) -> defaultdict:
        if all([s in TONE_DIACRITICS_MAP for s in supraseg]):
            tone_eq = ''.join([TONE_DIACRITICS_MAP[s] for s in supraseg])
            return self.get_tonal_features(tone_eq)
        else:
            features = defaultdict(lambda:0)
            for s in supraseg:
                for feature, value in DIACRITICS_EFFECTS[s]:
                    features[feature] = max(value, features[feature])
            return features

    def get_manner(self) -> str:
        if self.phone_class in ('CONSONANT', 'GLIDE'):
            if self.base in AFFRICATES or segment_is_affricate(self.segment):
                manner = 'AFFRICATE'
            elif self.base in PLOSIVES:
                manner = 'PLOSIVE'
            elif self.base in NASALS:
                manner = 'NASAL'
            elif self.base in FRICATIVES and '̞' in self.segment: # lowered diacritic turns fricatives into approximants
                manner = 'APPROXIMANT'
            elif re.search('[ɬɮ]', self.base):
                manner = 'LATERAL FRICATIVE'
            elif self.base in FRICATIVES:
                manner = 'FRICATIVE'
            elif self.base in TRILLS and '̝' in self.segment:
                manner = 'FRICATIVE TRILL'
            elif self.base in TRILLS:
                manner = 'TRILL'
            elif self.base in TAPS_AND_FLAPS:
                manner = 'TAP/FLAP'
            elif self.features['lateral'] == 1 and self.features['approximant'] == 1:
                manner = 'LATERAL APPROXIMANT'
            elif self.base in APPROXIMANTS:
                manner = 'APPROXIMANT'
            elif self.base in IMPLOSIVES:
                manner = 'IMPLOSIVE'
            elif self.base in CLICKS:
                manner = 'CLICK'
            else:
                raise ValueError(f'Could not determine manner of articulation for {self.segment}')
        else:
            manner = self.phone_class

        # Add features marked only by diacritics
        if re.search('̚', self.segment):
            manner = 'UNRELEASED ' + manner
        elif re.search('ˡ', self.segment):
            manner = 'LATERAL RELEASED ' + manner
        if re.search('̃', self.segment):
            manner = 'NASALIZED ' + manner
        elif re.match(r'[ᵐᶬⁿᵑ]', self.segment):
            manner = 'PRENASALIZED ' + manner
        elif re.search(r'.+[ᵐᶬⁿᵑ]', self.segment):
            manner = 'NASAL RELEASED ' + manner
        if re.match(r'[ʰʱ]', self.segment):
            manner = 'PREASPIRATED ' + manner
        elif re.search(r'[ʰʱ]', self.segment):
            manner = 'ASPIRATED ' + manner
        if re.search('ᶣ|(ʲʷ)|(ʷʲ)', self.segment):
            manner = 'LABIO-PALATALIZED ' + manner
        elif re.search('ʲ', self.segment):
            manner = 'PALATALIZED ' + manner
        elif re.search('ʷ', self.segment):
            manner = 'LABIALIZED ' + manner
        if re.search('˞', self.segment):
            manner = 'RHOTACIZED ' + manner
        if re.search('ˤ', self.segment):
            manner = 'PHARYNGEALIZED ' + manner
        elif re.search('ˠ', self.segment):
            manner = 'VELARIZED ' + manner
        elif re.search('ˀ', self.segment):
            manner = 'GLOTTALIZED ' + manner
        elif re.search('̤', self.segment):
            manner = 'BREATHY ' + manner
        elif re.search('̰', self.segment):
            manner = 'CREAKY ' + manner
        if re.search('͈', self.segment):
            manner = 'FORTIS ' + manner
        elif re.search('͉', self.segment):
            manner = 'LENIS ' + manner
        if re.search('˭', self.segment):
            manner = 'TENSE ' + manner
        if re.search('ʼ', self.segment):
            manner = 'EJECTIVE ' + manner
        if re.search('ˈ', self.segment):
            manner = 'STRESSED ' + manner
        elif re.search('ˌ', self.segment):
            manner = 'SECONDARY STRESSED ' + manner
        if re.search('ː', self.segment):
            manner = 'LONG ' + manner
        elif re.search('ˑ', self.segment):
            manner = 'HALF-LONG ' + manner
        elif re.search('̆', self.segment):
            manner = 'EXTRA SHORT ' + manner
        if re.search(r'[̩̍]', self.segment):
            manner = 'SYLLABIC ' + manner
        elif re.search('̯', self.segment) and self.phone_class == 'VOWEL':
            manner = 'NON-SYLLABIC ' + manner

        return manner

    def get_poa(self) -> str:
        val_err = ValueError(f'Could not determine place of articulation for {self.segment}')
        if self.phone_class in ('CONSONANT', 'GLIDE'):
            if re.search(r'([wʍ])|([kɡ].*͡[pb])', self.segment):
                return 'LABIAL-VELAR'
            elif re.search('̼', self.segment):
                return 'LINGUO-LABIAL'
            elif self.base in BILABIALS:
                return 'BILABIAL'
            elif self.features['labiodental'] == 1:
                return 'LABIODENTAL'
            elif re.search(r'[θðǀ̪]', self.segment):
                return 'DENTAL'
            elif re.search('̺', self.segment) and self.base in ALVEOLARS:
                return 'APICO-ALVEOLAR'
            elif re.search('̻', self.segment) and self.base in ALVEOLARS:
                return 'LAMINAL ALVEOLAR'
            elif self.base in ALVEOLARS:
                return 'ALVEOLAR'
            elif self.base in LATERALS:
                return 'LATERAL'
            elif self.base in POSTALVEOLARS:
                return 'POSTALVEOLAR'
            elif self.base in ALVEOLOPALATALS:
                return 'ALVEOLOPALATAL'
            elif self.base in RETROFLEXES:
                return 'RETROFLEX'
            elif self.base in PALATALS:
                return 'PALATAL'
            elif self.base in VELARS:
                return 'VELAR'
            elif self.base in UVULARS:
                return 'UVULAR'
            elif self.base in PHARYNGEALS:
                return 'PHARYNGEAL'
            elif self.base in EPIGLOTTALS:
                return 'EPIGLOTTAL'
            elif self.base in GLOTTALS:
                return 'GLOTTAL'
            else:
                raise val_err
            
        elif self.phone_class == 'VOWEL':
            # Height / Openness
            if CLOSE_VOWEL_REGEX.search(self.base):
                height = 'CLOSE'
            elif CLOSE_MID_VOWEL_REGEX.search(self.base):
                height = 'CLOSE-MID'
            elif self.base in {'ə', 'ɚ'}:
                height = 'MID'
            elif OPEN_MID_VOWEL_REGEX.search(self.base):
                height = 'OPEN-MID'
            elif OPEN_VOWEL_REGEX.search(self.base):
                height = 'OPEN'
            else:
                raise val_err
            
            # Frontness / Backness
            if FRONT_VOWEL_REGEX.search(self.base):
                frontness = 'FRONT'
            elif CENTRAL_VOWEL_REGEX.search(self.base):
                frontness = 'CENTRAL'
            elif BACK_VOWEL_REGEX.search(self.base):
                frontness = 'BACK'
            else:
                raise val_err
            
            # TODO add rounded to manner

            return ' '.join([height, frontness])

        elif self.phone_class == 'DIPHTHONG': # TODO add better description for diphthongs
            return ''

        elif self.phone_class == 'TONEME':
            # Level tones
            if self.features['tone_contour'] == 0:
                # Extra high level tone
                if self.features['tone_high'] == 1 and self.features['tone_central'] == 0:
                    return 'EXTRA HIGH LEVEL TONE'
                
                # High level tone
                elif self.features['tone_high'] == 1:
                    return 'HIGH LEVEL TONE'
                
                # Mid level tone
                elif self.features['tone_mid'] == 1:
                    return 'MID LEVEL TONE'

                # Low level tone
                elif self.features['tone_high'] == 0 and self.features['tone_central'] == 1:
                    return 'LOW LEVEL TONE'
                
                # Extra low level tone
                elif self.features['tone_high'] == 0 and self.features['tone_central'] == 0:
                    return 'EXTRA LOW LEVEL TONE'
                
                else:
                    raise val_err

            # Contour tones
            else:
                if self.features['tone_rising'] == 1 and self.features['tone_falling'] == 1:
                    if self.features['tone_convex'] == 1:
                        contour = 'RISING-FALLING'
                    else:
                        contour = 'FALLING-RISING'
                elif self.features['tone_rising'] == 1:
                    contour = 'RISING'
                elif self.features['tone_falling'] == 1:
                    contour = 'FALLING'
                else:
                    raise val_err
                
                # Distinguish high/low falling/rising
                if self.features['tone_high'] == 1 and self.features['tone_central'] == 1:
                    contour = 'HIGH ' + contour
                elif self.features['tone_high'] == 0 and self.features['tone_central'] == 1:
                    contour = 'LOW ' + contour

                return contour
        
        elif self.phone_class == 'SUPRASEGMENTAL': # TODO add better description for suprasegmentals
            return ''
            
        else:
            raise val_err

    def get_sonority(self) -> int:
        """
        Returns the sonority level of a segment according to Parker's (2002) universal sonority hierarchy.
        Determines appropriate sonority level for a segment by checking membership in phonological classes and/or relevant features.
        """
        from .sonority import SONORITY_LEVELS

        # Vowels and diphthongs
        if self.phone_class in ('VOWEL', 'DIPHTHONG'):

            # Check if diphthong
            if self.phone_class == 'DIPHTHONG':
                # Diphthong: calculate sonority as maximum sonority of component parts
                diphthong_components = [Segment(seg) for seg in segment_ipa(self.segment, combine_diphthongs=False)]
                return max([seg.sonority for seg in diphthong_components])

            # Treat as glide if non-syllabic
            if self.features['syllabic'] == 0:
                return SONORITY_LEVELS["GLIDES"]
            
            # Schwa /ə/ and /ɨ/ have special sonority
            elif self.base == 'ə':
                return SONORITY_LEVELS["/ə/"]
            elif self.base == 'ɨ':
                return SONORITY_LEVELS["/ɨ/"]
            
            # Open and near-open vowels
            elif ((self.features['high'] == 0) and (self.features['low'] == 1)): 
                return SONORITY_LEVELS["OPEN VOWELS"]
            
            # Open-mid, mid, close-mid vowels other than schwa /ə/
            elif self.features['high'] == 0:  
                return SONORITY_LEVELS["MID VOWELS"]
            
            # Near-close and close vowels other than /ɨ/
            elif self.features['high'] == 1:
                return SONORITY_LEVELS["CLOSE VOWELS"]

        # Glides
        elif self.phone_class == 'GLIDE':
            return SONORITY_LEVELS["GLIDES"]

        # Consonants
        elif self.phone_class == 'CONSONANT':
            
            # Non-glide, non-lateral, non-tap/flap/trill approximants: /ʋ, ɹ, ɻ, R/
            if self.features['approximant'] == 1 and self.features['lateral'] == 0 and self.features['trill'] == 0 and self.features['tap'] == 0:
                return SONORITY_LEVELS["GENERAL APPROXIMANTS"]
            
            # Lateral approximants
            elif self.manner == 'LATERAL APPROXIMANT':
                return SONORITY_LEVELS["LATERAL APPROXIMANTS"]
            
            # Taps/flaps
            elif self.manner == 'TAP/FLAP':
                return SONORITY_LEVELS["TAPS"]
            
            # Trills
            elif self.manner == 'TRILL':
                return SONORITY_LEVELS["TRILLS"]
            
            # Nasals
            elif self.manner == 'NASAL':
                return SONORITY_LEVELS["NASALS"]
            
            # /h/
            elif self.base == 'h':
                return SONORITY_LEVELS["/h/"]
            
            # Fricatives
            elif re.search('FRICATIVE', self.manner):
                
                # Voiced fricatives
                if self.features['periodicGlottalSource'] == 1:
                    return SONORITY_LEVELS["VOICED FRICATIVES"]
                
                # Voiceless fricatives
                else:
                    return SONORITY_LEVELS["VOICELESS FRICATIVES"]
            
            # Affricates, plosives, implosives, clicks
            else:
            
                # Voiced
                if self.features['periodicGlottalSource'] == 1:
                    return SONORITY_LEVELS["VOICED PLOSIVES"]
                    
                # Voiceless 
                else:
                    return SONORITY_LEVELS["VOICELESS PLOSIVES"]
        
        # Tonemes and other suprasegmentals
        elif self.phone_class in ('TONEME', 'SUPRASEGMENTAL'):
            return SONORITY_LEVELS["TONEME"] # Highest sonority

        # Other sounds: raise error message
        raise ValueError(f'Error: the sonority of phone "{self.segment}" cannot be determined!')

    def __str__(self) -> str:
        """Print the segment and its class, manner, place of articulation, sonority, and/or other relevant features."""
        info = [
            f'/{self.segment}/',
            f'Class: {self.phone_class}'
        ]

        if self.phone_class in ('CONSONANT', 'VOWEL', 'GLIDE'):
            info.append(f'Place of Articulation: {self.poa}')
        if self.phone_class in ('CONSONANT', 'VOWEL', 'GLIDE', 'DIPHTHONG'):
            info.extend([
                f'Manner of Articulation: {self.manner}',
                f'Voiced: {self.voiced is True}',
                f'Sonority: {self.sonority}',
            ])
        
        elif self.phone_class == 'TONEME':
            shape = re.search(r'((RISING|FALLING|LEVEL)-?(RISING|FALLING)?)', self.poa).group()
            info.append(f'Shape: {shape}')
            level = re.search(r'((EXTRA )?(HIGH|MID|LOW)-?(MID)?)', self.poa)
            if level:
                level = level.group()
                info.append(f'Level: {level}')

        return '\n'.join(info)
        

# IPA STRING SEGMENTATION
def segment_ipa(word,
                remove_ch: set = None,
                combine_diphthongs: bool = True,
                preaspiration: bool = True,
                autonomous_diacritics: set = None
                ):
    """Returns a list of segmented phones from the word"""

    # Assert that all characters in string are recognized IPA characters
    try:
        verify_charset(word)
    except ValueError:
        word = normalize_ipa_ch(word)
        verify_charset(word)
    
    # Remove spaces and other specified characters/diacritics (e.g. stress, linking ties for phonological words)
    if remove_ch is None:
        remove_ch = set()
    remove_ch.update('‿')  # liaison tie
    word = re.sub(fr"[{remove_ch}\s]", '', word)

    # Split by inter-diacritics, which don't seem to match properly in regex
    parts = re.split('͡|͜', word)

    # Then segment the parts and re-combine with tie character, as necessary
    segments = []
    for part in parts:
        segmented = SEGMENT_REGEX.findall(part)
        if len(segments) > 0:
            segments[-1] = ''.join([segments[-1], '͡', segmented[0]])
            segments.extend(segmented[1:])
        else:
            segments.extend(segmented)
    
    # For some reason toneme segmentation by regex is a bit glitchy and adjacent components of tonemes are segmented separately
    # even if an appropriate regex is supplied; something because of the diacritic characters not interacting well with regex
    # e.g. 'jɐn⁵⁵' -> ['j', 'ɐ', 'n', '⁵', '⁵'] rather than ['j', 'ɐ', 'n', '⁵⁵']
    # Combine these back together
    # TODO add pytests to confirm that this segmentation works
    if TONEME_REGEX.search(word):
        toneme_i = [i for i, seg in enumerate(segments) if TONEME_REGEX.search(seg)]
        if len(toneme_i) > 1:
            toneme_i.reverse()
            for i, index in enumerate(toneme_i):
                try:
                    prev_index = toneme_i[i+1] # previous toneme index is next in list, since it is reversed
                    if index-1 == prev_index:
                        segments[prev_index] += segments[index]
                        segments = [s for i, s in enumerate(segments) if i != index]
                except IndexError:
                    pass

    # Move aspiration diacritic <ʰʱ> from preceding vowel or glide to following consonant
    # Can't easily be distinguished in regex since the same symbol is usually a post-diacritic for post-aspiration
    if preaspiration:
        for i, seg in enumerate(segments):
            preasp = PREASPIRATION_REGEX.search(seg)
            if preasp:
                try:
                    match = preasp.group()
                    segments[i] = re.sub('[ʰʱ]$', '', seg)
                    segments[i+1] = match + segments[i+1]
                except KeyError:
                    pass

    # Combine diphthongs
    if combine_diphthongs:
        updated_segments = []
        i = 0
        while i < len(segments):
        #for i, seg in enumerate(segments):
            seg = segments[i]
            if '̯' in seg and segment_is_vowel(seg):
                if i > 0:
                    # First try to combine with preceding vowel
                    if segment_is_vowel(updated_segments[-1]):
                        updated_segments[-1] += seg
                        i += 1

                    # If there is no suitable preceding vowel, try combining with following vowel instead
                    elif segment_is_vowel(segments[i+1]):
                        updated_segments.append(seg+segments[i+1])
                        i += 2

                    # Else do nothing
                    else:
                        updated_segments.append(seg)
                        i += 1

                # Combine an initial non-syllabic vowel onto a following vowel
                else:
                    if segment_is_vowel(segments[1]):
                        updated_segments.append(seg+segments[1])
                        i += 2
            else:
                updated_segments.append(seg)
                i += 1
        
        segments = updated_segments
    
    # Split off specified free-standing/autosegmental (typically prosodic) units (or other diacritics)
    if autonomous_diacritics:
        autonomous_diacritic_regex = re.compile(rf'[{autonomous_diacritics}]')
        updated_segments = []
        for segment in segments:
            matches = autonomous_diacritic_regex.findall(segment)
            if matches:
                seg_minus_diacritic = autonomous_diacritic_regex.sub('', segment)
                if seg_minus_diacritic:
                    updated_segments.append(seg_minus_diacritic)
                updated_segments.append(''.join(matches))
            else:
                updated_segments.append(segment)
        segments = updated_segments

    return segments


# Helper functions for identifying phones with particular features
def lookup_segments(feature_values: dict,
                    segment_list: Iterable = IPA_SEGMENTS
                    ) -> set:
    """Returns a list of segments whose feature values match the search criteria"""
    matches = []
    for segment in segment_list:
        segment = Segment(segment)
        match_tallies = 0
        for feature, value in feature_values.items():
            if segment.features[feature] == value:
                match_tallies += 1
        if match_tallies == len(feature_values):
            matches.append(segment.segment)
    return set(matches)


def common_features(segment_list: Iterable,
                    start_features: Iterable = FEATURE_SET):
    """Returns the features/values shared by all segments in the list"""
    features = set(start_features)
    feature_values = defaultdict(lambda:[])
    for segment in segment_list:
        segment = Segment(segment)
        for feature in features:
            value = segment.features[feature]
            if value not in feature_values[feature]:
                feature_values[feature].append(value)
    common = [(feature, feature_values[feature][0]) for feature in feature_values if len(feature_values[feature]) == 1]
    return common
