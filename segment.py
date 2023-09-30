
import os
import re
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phonUtils.initPhoneData import (
    # Top-level phone sets
    vowels, glides, consonants, tonemes,
    # Phone classes by manner of articulation
    plosives, implosives, nasals, affricates, fricatives, trills, taps_flaps, approximants, glides, clicks,
    # Phone classes by place of articulation
    bilabial, alveolar, lateral, postalveolar, alveolopalatal, retroflex, palatal, velar, uvular, pharyngeal, epiglottal, glottal,
    # Diacritics and associated features
    diacritics_effects,
    # Phonological features and feature geometry weights 
    phone_features, tone_levels,
    # IPA regexes and constants
    segment_regex, preaspiration_regex, diphthong_regex, suprasegmental_diacritics, toneme_regex,
    front_vowel_regex, central_vowel_regex, back_vowel_regex, 
    close_vowel_regex, close_mid_vowel_regex, open_vowel_regex, open_mid_vowel_regex,
    # Helper functions
    _is_affricate
)
from phonUtils.ipaTools import strip_diacritics, normalize_ipa_ch, verify_charset

# TODO make toneme-tone diacritic map file
tone_diacritics_map = {
            '̏':'˩',
            '̀':'˨',
            '̄':'˧',
            '́':'˦',
            '̋':'˥',
            '̂':'˥˩',
            '̌':'˩˥',
            '᷅':'˩˨',
            '᷄':'˦˥',
            '᷈':'˧˦˧',
            }


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
        
        # Get phone class
        self.phone_class = self.get_phone_class()

        # Get voicing status, manner, and place of articulation
        if self.phone_class not in ('TONEME', 'SUPRASEGMENTAL'):
            self.voiced = self.features['periodicGlottalSource'] == 1
            self.voiceless = self.features['periodicGlottalSource'] == 0
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
            #raise ValueError(f'Error: invalid segment <{self.segment}>, no base IPA character found!')
            return '', self.segment
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
        elif self.base in tonemes or self.base in tone_diacritics_map:
            return 'TONEME'
        elif all([ch in diacritics_effects for ch in self.base]):
            return 'SUPRASEGMENTAL'
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
                    #raise AssertionError(f'Error: invalid segment <{segment}>, no base IPA character found!')
                    return self.get_suprasegmental_features(segment)
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
        toneme_id = defaultdict(lambda:0)
        for feature in phone_features[base]:
            toneme_id[feature] = phone_features[base][feature]
        
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


    def get_suprasegmental_features(self, supraseg):
        if all([s in tone_diacritics_map for s in supraseg]):
            tone_eq = ''.join([tone_diacritics_map[s] for s in supraseg])
            return self.get_tonal_features(tone_eq)
        else:
            features = defaultdict(lambda:0)
            for s in supraseg:
                for feature, value in diacritics_effects[s]:
                    features[feature] = max(value, features[feature])
            return features


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
        
        elif self.phone_class == 'SUPRASEGMENTAL': # TODO add better description for suprasegmentals
            return ''
            
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
        
        # Tonemes and other suprasegmentals
        elif self.phone_class in ('TONEME', 'SUPRASEGMENTAL'):
            return 17 # Highest sonority
        
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
        
        elif self.phone_class == 'TONEME':
            shape = re.search(r'((RISING|FALLING|LEVEL)-?(RISING|FALLING)?)', self.poa).group()
            info.append(f'Shape: {shape}')
            level = re.search(r'((EXTRA )?(HIGH|MID|LOW)-?(MID)?)', self.poa)
            if level:
                level = level.group()
                info.append(f'Level: {level}')

        return '\n'.join(info)
        

# IPA STRING SEGMENTATION
def segment_ipa(word, remove_ch='', combine_diphthongs=True, preaspiration=True, suprasegmentals=None):
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
    
    # For some reason toneme segmentation by regex is a bit glitchy and adjacent components of tonemes are segmented separately
    # even if an appropriate regex is supplied; something because of the diacritic characters not interacting well with regex
    # e.g. 'jɐn⁵⁵' -> ['j', 'ɐ', 'n', '⁵', '⁵'] rather than ['j', 'ɐ', 'n', '⁵⁵']
    # Combine these back together
    # TODO add pytests to confirm that this segmentation works
    if toneme_regex.search(word):
        toneme_i = [i for i, seg in enumerate(segments) if toneme_regex.search(seg)]
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
    
    # Split off suprasegmentals
    if suprasegmentals:
        suprasegmental_regex = re.compile(rf'[{suprasegmental_diacritics}{tonemes}{suprasegmentals}]')
        updated_segments = []
        for segment in segments:
            matches = suprasegmental_regex.findall(segment)
            if matches:
                updated_segments.append(''.join(matches))
                seg_minus_supraseg = suprasegmental_regex.sub('', segment)
                if seg_minus_supraseg:
                    updated_segments.append(seg_minus_supraseg)
            else:
                updated_segments.append(segment)
        segments = updated_segments

    return segments


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

