# PHONETIC SEGMENT ANALYSIS AND PHONETIC SIMILARITY/DISTANCE
# Code developed by Philip Georgis (Last updated: August 2023)

import os
import re
import sys
from collections import defaultdict
from functools import lru_cache
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cosine

# Add the project's root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load phonological constants initialized in initPhoneData.py
from PhoneticSimilarity.initPhoneData import (
    # Top-level phone sets
    vowels, glides, consonants, tonemes,
    # Phone classes by manner of articulation
    plosives, implosives, nasals, affricates, fricatives, trills, taps_flaps, liquids, rhotics, approximants, glides, clicks,
    # Phone classes by place of articulation
    bilabial, labiodental, dental, alveolar, lateral, postalveolar, alveolopalatal, retroflex, palatal, velar, uvular, pharyngeal, epiglottal, glottal,
    # Diacritics and associated features
    diacritics, diacritics_effects, post_diacritics, suprasegmental_diacritics,
    # Phonological features and feature geometry weights 
    phone_features, feature_weights, tone_levels,
    # IPA regexes and constants
    segment_regex, preaspiration_regex, diphthong_regex, diacritic_regex,
    front_vowel_regex, central_vowel_regex, back_vowel_regex, 
    close_vowel_regex, close_mid_vowel_regex, open_vowel_regex, open_mid_vowel_regex,  
    # IPA character normalization/validation
    valid_ipa_ch, ipa_norm_map,
    # Helper functions
    _is_affricate
)

# FUNCTIONS FOR IPA STRING MANIPULATION AND NORMALIZATION
def strip_diacritics(string, excepted=[]):
    """Removes diacritic characters from an IPA string
    By default removes all diacritics; in order to keep certain diacritics,
    these should be passed as a list to the "excepted" parameter"""
    if len(excepted) > 0:
        to_remove = ''.join([d for d in diacritics if d not in excepted])
        return re.sub(f'[{to_remove}]', '', string)
    else:
        return diacritic_regex.sub('', string)


def normalize_ipa_ch(string, ipa_norm_map=ipa_norm_map):
    """Normalizes some commonly mistyped IPA characters according to a pre-loaded normalization mapping dictionary"""

    def replace_callback(match):
        return ipa_norm_map[match.group(0)]

    pattern = re.compile('|'.join(map(re.escape, ipa_norm_map.keys())))
    string = pattern.sub(replace_callback, string)

    return string


def invalid_ch(string, valid_ch=valid_ipa_ch):
    """Returns set of unrecognized (non-IPA) characters in phonetic string"""
    return set(re.findall(fr'[^{valid_ch}]', string))


def verify_charset(string):
    """Verifies that all characters are valid IPA characters or diacritics, otherwise raises error"""
    unk_ch = invalid_ch(string)
    if len(unk_ch) > 0:
        unk_ch_str = '>, <'.join(unk_ch)
        raise ValueError(f'Invalid IPA character(s) <{unk_ch_str}> found in "{string}"!')


# IPA STRING SEGMENTATION
def segment_ipa(word, remove_ch='', combine_diphthongs=True, preaspiration=True):
    """Returns a list of segmented phones from the word"""

    # Assert that all characters in string are recognized IPA characters
    verify_charset(word)
    
    # Remove spaces and other specified characters/diacritics (e.g. stress, linking ties for phonological words)
    remove_ch += '\s‿'
    word = re.sub(f"[{remove_ch}]", '', word)

    # Split by inter-diacritics, which don't seem to match properly in regex
    parts = re.split('͡|͜', word)

    # Then segment the parts and re-combine with tie character, as necessary
    segments = []
    for part in parts:
        segmented = segment_regex.findall(part)
        if len(segments) > 0:
            segments[-1] = ''.join([segments[-1], '͡', segmented[0]])
            segments.extend(segmented[1:])
        else:
            segments.extend(segmented)

    # Move aspiration diacritic <ʰʱ> from preceding vowel or glide to following consonant
    # Can't easily be distinguished in regex since the same symble is usually a post-diacritic for post-aspiration
    if preaspiration:
        for i, seg in enumerate(segments):
            preasp = preaspiration_regex.search(seg)
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
            if '̯' in seg and _is_vowel(seg):
                if i > 0:
                    # First try to combine with preceding vowel
                    if _is_vowel(updated_segments[-1]):
                        updated_segments[-1] += seg
                        i += 1

                    # If there is no suitable preceding vowel, try combining with following vowel instead
                    elif _is_vowel(segments[i+1]):
                        updated_segments.append(seg+segments[i+1])
                        i += 2

                    # Else do nothing
                    else:
                        updated_segments.append(seg)
                        i += 1

                # Combine an initial non-syllabic vowel onto a following vowel
                else:
                    if _is_vowel(segments[1]):
                        updated_segments.append(seg+segments[1])
                        i += 2
            else:
                updated_segments.append(seg)
                i += 1
        
        segments = updated_segments

    return segments


class Segment:
    segments = {}

    def __init__(self, segment, normalize=False):
        # Normalize IPA string input and check for non-IPA characters
        self.segment = segment
        if normalize:
            self.normalize()

        # Base segment: no diacritics; first element of diphthongs, affricates, or complex consonants
        self.stripped, self.base = self.get_base_ch()

        # Get distinctive phonological feature dictionary
        self.features = self.get_phone_features(self.segment)
        
        # Get voicing status
        self.voiced = self.features['periodicGlottalSource'] == 1
        self.voiceless = self.features['periodicGlottalSource'] == 0

        # Get phone class, manner, and place of articulation
        self.phone_class = self.get_phone_class()
        self.manner = self.get_manner()
        self.poa = self.get_poa()

        # Get sonority
        self.sonority = self.get_sonority()

        # Add Segment instance to Segment class attribute dictionary
        Segment.segments[self.segment] = self


    def normalize(self):
        self.segment = normalize_ipa_ch(self.segment)
        verify_charset(self.segment)


    def get_base_ch(self):
        no_diacritics = strip_diacritics(self.segment)
        if len(no_diacritics) < 1:
            raise ValueError(f'Error: invalid segment <{self.segment}>, no base IPA character found!')
        else:
            return no_diacritics, no_diacritics[0]


    def get_phone_class(self):
        if diphthong_regex.search(self.segment):
            return 'DIPHTHONG'
        elif self.base in glides:
            return 'GLIDE'
        elif self.base in vowels:
            return 'VOWEL'
        elif self.base in consonants:
            return 'CONSONANT'
        elif self.base in tonemes:
            return 'TONEME'
        else:
            raise ValueError(f'Could not determine phone class of {self.segment}')


    def get_phone_features(self, segment):
        """Returns a dictionary of distinctive phonological feature values for the segment"""
        
        # Retrieve saved feature dictionary if already generated for this segment
        if segment in Segment.segments:
            return Segment.segments[segment].features

        # Generate an empty phone feature dictionary with default values of 0
        feature_dict = dict.fromkeys(phone_features[next(iter(phone_features))], 0)

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
                    raise AssertionError(f'Error: invalid segment <{segment}>, no base IPA character found!')
                bases.append(base)

                # If the length of the base > 1, the segment is a diphthong (e.g. /e̯a/) or complex toneme (e.g. /˥˩/)
                # Filter out tonemes to handle diphthongs first
                if (len(base) > 1) and (base[0] not in tonemes):
                    return self.get_diphthong_features(segment)

                # Handle tonemes
                elif base[0] in tonemes:
                    return self.get_tonal_features(segment)

                # Otherwise, retrieve the base phone's features
                else:
                    base_id = phone_features[base]
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
            if bases[0] in plosives and bases[-1] in fricatives:
                feature_dict['delayedRelease'] = 1
                feature_dict['continuant'] = 0

        return feature_dict



    def apply_diacritics(self, base:str, base_features:dict, diacritics:set):
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
            for feature, value in diacritics_effects[modifier]:
                modified_features[feature] = value
                
            if modifier == '̞': # lowered diacritic: turns fricatives into approximants
                if base[0] in fricatives:
                    modified_features['approximant'] = 1
                    modified_features['consonantal'] = 0
                    modified_features['delayedRelease'] = 0
                    modified_features['sonorant'] = 1
            
            elif modifier == '̝': # raised diacritic
                # turn approximants/trills into fricativized approximants
                if base[0] in approximants.union(trills):
                    modified_features['delayedRelease'] = 1
                    
                # turn fricatives into plosives
                elif base[0] in fricatives:
                    modified_features['continuant'] = 0
                    modified_features['delayedRelease'] = 0
        
        return modified_features


    def get_diphthong_features(self, diphthong):
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


    def get_tonal_features(self, toneme):
        """Computes complex tonal features"""
        
        # Set the base as the first component of the toneme
        base = toneme[0]
        
        # Create copy of original feature dictionary, or else it modifies the source
        toneme_id = {feature:phone_features[base][feature] for feature in phone_features[base]}
        
        # Get the tone level of each tonal component of the toneme
        toneme_levels = [tone_levels[t] for t in toneme if t in tonemes]
        
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


    def get_manner(self):
        if self.phone_class in ('CONSONANT', 'GLIDE'):
            if self.base in affricates or _is_affricate(self.segment):
                manner = 'AFFRICATE'
            elif self.base in plosives:
                manner = 'PLOSIVE'
            elif self.base in nasals:
                manner = 'NASAL'
            elif self.base in fricatives and '̞' in self.segment: # lowered diacritic turns fricatives into approximants
                manner = 'APPROXIMANT'
            elif re.search('[ɬɮ]', self.base):
                manner = 'LATERAL FRICATIVE'
            elif self.base in fricatives:
                manner = 'FRICATIVE'
            elif self.base in trills and '̝' in self.segment:
                manner = 'FRICATIVE TRILL'
            elif self.base in trills:
                manner = 'TRILL'
            elif self.base in taps_flaps:
                manner = 'TAP/FLAP'
            elif self.features['lateral'] == 1 and self.features['approximant'] == 1:
                manner = 'LATERAL APPROXIMANT'
            elif self.base in approximants:
                manner = 'APPROXIMANT'
            elif self.base in implosives:
                manner = 'IMPLOSIVE'
            elif self.base in clicks:
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
        

    def get_poa(self):
        val_err = ValueError(f'Could not determine place of articulation for {self.segment}')
        if self.phone_class in ('CONSONANT', 'GLIDE'):
            if re.search(r'([wʍ])|([kɡ].*͡[pb])', self.segment):
                return 'LABIAL-VELAR'
            elif re.search('̼', self.segment):
                return 'LINGUO-LABIAL'
            elif self.base in bilabial:
                return 'BILABIAL'
            elif self.features['labiodental'] == 1:
                return 'LABIODENTAL'
            elif re.search(r'[θðǀ̪]', self.segment):
                return 'DENTAL'
            elif re.search('̺', self.segment) and self.base in alveolar:
                return 'APICO-ALVEOLAR'
            elif re.search('̻', self.segment) and self.base in alveolar:
                return 'LAMINAL ALVEOLAR'
            elif self.base in alveolar:
                return 'ALVEOLAR'
            elif self.base in lateral:
                return 'LATERAL'
            elif self.base in postalveolar:
                return 'POSTALVEOLAR'
            elif self.base in alveolopalatal:
                return 'ALVEOLOPALATAL'
            elif self.base in retroflex:
                return 'RETROFLEX'
            elif self.base in palatal:
                return 'PALATAL'
            elif self.base in velar:
                return 'VELAR'
            elif self.base in uvular:
                return 'UVULAR'
            elif self.base in pharyngeal:
                return 'PHARYNGEAL'
            elif self.base in epiglottal:
                return 'EPIGLOTTAL'
            elif self.base in glottal:
                return 'GLOTTAL'
            else:
                raise val_err
            
        elif self.phone_class == 'VOWEL':
            # Height / Openness
            if close_vowel_regex.search(self.base):
                height = 'CLOSE'
            elif close_mid_vowel_regex.search(self.base):
                height = 'CLOSE-MID'
            elif self.base in {'ə', 'ɚ'}:
                height = 'MID'
            elif open_mid_vowel_regex.search(self.base):
                height = 'OPEN-MID'
            elif open_vowel_regex.search(self.base):
                height = 'OPEN'
            else:
                raise val_err
            
            # Frontness / Backness
            if front_vowel_regex.search(self.base):
                frontness = 'FRONT'
            elif central_vowel_regex.search(self.base):
                frontness = 'CENTRAL'
            elif back_vowel_regex.search(self.base):
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

        else:
            raise val_err


    def get_sonority(self):
        """Returns the sonority level of a phone according to Parker's (2002) 
        universal sonority hierarchy
        
        adapted from:
        https://www.researchgate.net/publication/336652515/figure/fig1/AS:815405140561923@1571419143959/Adapted-version-of-Parkers-2002-sonority-hierarchy.ppm
        """
        # Determine appropriate sonority level by checking membership in sound 
        # groups (manner/place of articulation) and/or relevant features     

        # Vowels
        if self.phone_class in ('VOWEL', 'DIPHTHONG', 'GLIDE'):

            # Check if diphthong
            if self.phone_class == 'DIPHTHONG':
                # Diphthong: calculate sonority as maximum sonority of component parts
                diphthong_components = [_toSegment(seg) for seg in segment_ipa(self.segment, combine_diphthongs=False)]
                return max([seg.get_sonority() for seg in diphthong_components])

            # Treat as glide if non-syllabic
            if self.features['syllabic'] == 0:
                return 11
            
            # Schwa /ə/ and /ɨ/ have special sonority
            elif self.base == 'ə':
                return 13
            elif self.base == 'ɨ':
                return 12
            
            # Open and near-open vowels
            elif ((self.features['high'] == 0) and (self.features['low'] == 1)): 
                return 16
            
            # Open-mid, mid, close-mid vowels other than schwa /ə/
            elif self.features['high'] == 0:  
                return 15
            
            # Near-close and close vowels other than /ɨ/
            elif self.features['high'] == 1:
                return 14

        # Glides
        elif self.phone_class == 'GLIDE':
            return 11
        
        # Consonants
        elif self.phone_class == 'CONSONANT':
            
            # Non-glide, non-lateral, non-tap/flap/trill approximants: /ʋ, ɹ, ɻ, R/
            if self.features['approximant'] == 1 and self.features['lateral'] == 0 and self.features['trill'] == 0 and self.features['tap'] == 0:
                return 10
            
            # Lateral approximants
            elif self.manner == 'LATERAL APPROXIMANT':
                return 9
            
            # Taps/flaps
            elif self.manner == 'TAP/FLAP':
                return 8
            
            # Trills
            elif self.manner == 'TRILL':
                return 7
            
            # Nasals
            elif self.manner == 'NASAL':
                return 6
            
            # /h/
            elif self.base == 'h':
                return 5
            
            # Fricatives
            elif re.search('FRICATIVE', self.manner):
                
                # Voiced fricatives
                if self.features['periodicGlottalSource'] == 1:
                    return 4
                
                # Voiceless fricatives
                else:
                    return 3
            
            # Affricates, plosives, implosives, clicks
            else:
            
                # Voiced
                if self.features['periodicGlottalSource'] == 1:
                    return 2
                    
                # Voiceless 
                else:
                    return 1
        
        # Tonemes
        elif self.phone_class == 'TONEME':
            return 0
        
        # Other sounds: raise error message
        else:
            raise ValueError(f'Error: the sonority of phone "{self.segment}" cannot be determined!')


    def __str__(self):
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
        
        else: # TONEME
            shape = re.search(r'((RISING|FALLING|LEVEL)-?(RISING|FALLING)?)', self.poa).group()
            info.append(f'Shape: {shape}')
            level = re.search(r'((EXTRA )?(HIGH|MID|LOW)-?(MID)?)', self.poa)
            if level:
                level = level.group()
                info.append(f'Level: {level}')

        return '\n'.join(info)
        

# AUXILIARY FUNCTIONS
def _toSegment(ch):
    return Segment.segments.get(ch, Segment(ch))      


def _is_ch(ch, l):
    try:
        if strip_diacritics(ch)[0] in l:
            return True
        else:
            return False
    except IndexError:
        return False


def _is_vowel(ch):
    return _is_ch(ch, vowels)


def _is_front_env(ch):
    if front_vowel_regex.search(ch) or ch in {'j', 'ɥ'}:
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
        return 'T'

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
                env = '#SS'
            elif sonority_i == next_sonority:
                env = '#S='
            elif sonority_i > next_sonority:
                env = '#S>'
            else: # sonority_i < next_sonority:
                env = '#S<'
        
            # Add front vowel environment
            if _is_front_env(next_segment.base):
                env += 'F'
                    
            return env
        
        # Free-standing segments
        else:
            return '#S#'
    
    # Word-final segments: S#
    elif i == len(segments)-1:
        assert len(segments) > 1
        prev_segment = segments[i-1]
        if prev_segment == segment_i:
            env = 'SS#'
        else:
            prev_sonority, sonority_i = prev_segment.sonority, segment_i.sonority
            
            if prev_sonority == sonority_i:
                env = '=S#'

            elif prev_sonority < sonority_i:
                env = '<S#'

            else: # prev_sonority > sonority_i
                env = '>S#' 

        # Add front vowel environment
        if _is_front_env(prev_segment.base):
            env = 'F' + env
        
        return env
    
    # Word-medial segments
    else:
        prev_segment, next_segment = segments[i-1], segments[i+1]
        prev_sonority, sonority_i, next_sonority = prev_segment.sonority, segment_i.sonority, next_segment.sonority
        
        # Sonority plateau: =S=
        if prev_segment == sonority_i == next_sonority:
            env = '=S='
        
        # Sonority peak: <S>
        elif prev_sonority < sonority_i > next_sonority:
            env = '<S>'
        
        # Sonority trench: >S< # TODO is this the best term?
        elif prev_sonority > sonority_i < next_sonority:
            env = '>S<'
        
        # Descending sonority: >S>
        elif prev_sonority > sonority_i > next_sonority:
            env = '>S>'
        
        # Ascending sonority: <S<
        elif prev_sonority < sonority_i < next_sonority:
            env = '<S<'
        
        elif prev_sonority < sonority_i == next_sonority:
            if segment_i == next_segment:
                env = '<SS'
            else:
                env = '<S='
        
        elif prev_sonority > sonority_i == next_sonority:
            if segment_i == next_segment:
                env = '>SS'
            else:
                env = '>S='
        
        elif prev_sonority == sonority_i < next_sonority:
            if segment_i == prev_segment:
                env = 'SS<'
            else:
                env = '=S<'
        
        elif prev_sonority == sonority_i > next_sonority:
            if segment_i == prev_segment:
                env = 'SS>'
            else:
                env = '=S>'
        
        elif prev_sonority == sonority_i == next_sonority:
            if segment_i == prev_segment:
                env = 'SS='
            elif segment_i == next_segment:
                env = '=SS'
            else:
                env = '=S='
        
        else:
            raise ValueError(f'Unable to determine environment for segment {i} /{segments[i].segment}/ within /{"".join([seg.segment for seg in segments])}/')
        
        # Add front vowel environment
        if _is_front_env(prev_segment.base):
            env = 'F' + env
        if _is_front_env(next_segment.base):
            env += 'F'
        
        return env

# SIMILARITY / DISTANCE MEASURES
def hamming_distance(vec1, vec2, normalize=True):
    differences = len([feature for feature in vec1 if vec1[feature] != vec2[feature]])
    if normalize:
        return differences / len(vec1)
    else: 
        return differences


def jaccard_sim(vec1, vec2):
    features = sorted(list(vec1.keys()))
    vec1_values = [vec1[feature] for feature in features]
    vec2_values = [vec2[feature] for feature in features]
    
    # Jaccard index does not allow continuous features
    # Ensure that they are all binarily encoded (any continuous value >0 --> 1)
    for vec in [vec1_values, vec2_values]:
        for i in range(len(vec)):
            if vec[i] > 0:
                vec[i] = 1
                
    return jaccard_score(vec1_values, vec2_values)


def dice_sim(vec1, vec2):
    jaccard = jaccard_sim(vec1, vec2)
    return (2*jaccard) / (1+jaccard)


def weighted_hamming(vec1, vec2, weights=feature_weights):
    diffs = 0
    for feature in vec1:
        if vec1[feature] != vec2[feature]:
            diffs += weights[feature]
    return diffs/len(vec1)


def weighted_jaccard(vec1, vec2, weights=feature_weights):
    union, intersection = 0, 0
    for feature in vec1:
        if ((vec1[feature] == 1) and (vec2[feature] == 1)):
            intersection += weights[feature]
        if ((vec1[feature] == 1) or (vec2[feature] == 1)):
            union += weights[feature]
    return intersection/union
            

def weighted_dice(vec1, vec2, weights=feature_weights):
    w_jaccard = weighted_jaccard(vec1, vec2, weights)
    return (2*w_jaccard) / (1+w_jaccard)


# PHONE COMPARISON
@lru_cache(maxsize=None)
def phone_sim(phone1, phone2, similarity='weighted_dice', exclude_features=None):
    """Returns the similarity of the features of the two phones according to
    the specified distance/similarity function;
    Features not to be included in the comparison should be passed as a list to
    the exclude_features parameter (by default no features excluded)"""

    if exclude_features is None:
        exclude_features = set()

    # Convert IPA strings to Segment objects
    phone1, phone2 = map(_toSegment, [phone1, phone2])

    # Get feature dictionaries for each phone
    phone_id1, phone_id2 = phone1.features.copy(), phone2.features.copy()

    # Exclude specified features
    for feature in exclude_features:
        phone_id1.pop(feature, None)
        phone_id2.pop(feature, None)

    # Calculate similarity of phone features according to specified measure
    if similarity in ['cosine', 'weighted_cosine']:
        compare_features = list(phone_id1.keys())
        phone1_values = [phone_id1[feature] for feature in compare_features]
        phone2_values = [phone_id2[feature] for feature in compare_features]
        if similarity == 'weighted_cosine':
            weights = [feature_weights[feature] for feature in compare_features]
            # Subtract from 1: cosine() returns a distance
            score = 1 - cosine(phone1_values, phone2_values, w=weights)
        else:
            score = 1 - cosine(phone1_values, phone2_values)
    else:
        measure = {
            'dice': dice_sim,
            'hamming': hamming_distance,
            'jaccard': jaccard_sim,
            'weighted_dice': weighted_dice,
            'weighted_hamming': weighted_hamming,
            'weighted_jaccard': weighted_jaccard
        }.get(similarity)
        if measure is None:
            raise KeyError(f'Error: similarity measure "{similarity}" not recognized!')
        score = measure(phone_id1, phone_id2)

    # If method is Hamming, convert distance to similarity
    if similarity in ['hamming', 'weighted_hamming']:
        score = 1 - score

    return score


# Helper functions for identifying phones with particular features
def lookup_segments(features, values, segment_list=consonants.union(vowels).union(tonemes)):
    """Returns a list of segments whose feature values match the search criteria"""
    matches = []
    for segment in segment_list:
        segment = _toSegment(segment)
        match_tallies = 0
        for feature, value in zip(features, values):
            if segment.features[feature] == value:
                match_tallies += 1
        if match_tallies == len(features):
            matches.append(segment.segment)
    return set(matches)


def common_features(segment_list, start_features=feature_weights.keys()):
    """Returns the features/values shared by all segments in the list"""
    features = set(start_features)
    feature_values = defaultdict(lambda:[])
    for segment in segment_list:
        segment = _toSegment(segment)
        for feature in features:
            value = segment.features[feature]
            if value not in feature_values[feature]:
                feature_values[feature].append(value)
    common = [(feature, feature_values[feature][0]) for feature in feature_values if len(feature_values[feature]) == 1]
    return common