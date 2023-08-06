# PHONETIC SEGMENT ANALYSIS AND PHONETIC DISTANCE
# Code written by Philip Georgis (2021)

# LOAD REQUIRED PACKAGES AND FUNCTIONS
import re, math, os
import pandas as pd
from collections import defaultdict
from sklearn.metrics import jaccard_score
from scipy.spatial.distance import cosine

def strip_ch(string, to_remove):
    """Removes a set of characters from strings"""
    to_remove_regex = '|'.join(to_remove)
    string = re.sub(to_remove_regex, '', string)
    return string

# IMPORT PHONE DATA
save_dir = os.path.dirname(__file__)
phone_data = pd.read_csv(os.path.join(save_dir, 'Phones', 'segments.csv'), sep=',')

def binary_feature(feature):
    """Converts features of type ['0', '-', '+'] to binary [0, 1]"""
    if str(feature) == '+':
        return 1
    else:
        return 0

# Dictionary of basic phones with their phonetic features
phone_features = {phone_data['segment'][i]:{feature:binary_feature(phone_data[feature][i])
                                          for feature in phone_data.columns
                                          if feature not in ['segment', 'sonority']
                                          if not pd.isnull(phone_data[feature][i])}
                  for i in range(len(phone_data))}

features = set(feature for sound in phone_features for feature in phone_features[sound])


# Dictionary of basic phones with their sonority levels
phone_sonority = {phone_data['segment'][i]:int(phone_data['sonority'][i])
                  for i in range(len(phone_data))}

max_sonority = max(phone_sonority.values())


# Load basic groupings of phones; e.g. plosive, fricative, velar, palatal
phone_classes = pd.read_csv(os.path.join(save_dir, 'Phones/phone_classes.tsv'))
phone_classes = {phone_classes['Group'][i]:phone_classes['Phones'][i].split()
                for i in range(len(phone_classes))}

# Set these phone groups as global variables so that they are callable by name
globals().update(phone_classes)

# Set basic consonants and vowels using syllabic feature
consonants = set(phone for phone in phone_features
              if phone_features[phone]['syllabic'] == 0
              if phone not in tonemes)
vowels = set(phone for phone in phone_features if phone not in consonants.union(tonemes))

# All basic sounds
all_sounds = ''.join(consonants.union(vowels).union(tonemes))

# IMPORT DIACRITICS DATA
diacritics_data = pd.read_csv(os.path.join(save_dir, 'Phones', 'diacritics.tsv'), sep='\t')

# Create dictionary of diacritic characters with affected features and values
diacritics_effects = defaultdict(lambda:[])
for i in range(len(diacritics_data)):
    effect = (diacritics_data['Feature'][i], binary_feature(diacritics_data['Value'][i]))
    
    # Skip diacritics which have no effect on features
    if type(effect[0]) != float:
        
        # Add to dictionary, with diacritic as key
        diacritics_effects[diacritics_data['Diacritic'][i]].append(effect)

# Isolate suprasegmental diacritics
suprasegmental_diacritics = set(diacritics_data.Diacritic[i] 
                                for i in range(len(diacritics_data)) 
                                if diacritics_data.Type[i] == 'suprasegmental')
suprasegmental_diacritics.remove('ː') # don't include length as a suprasegmental


# Diacritics by position with respect to base segments
inter_diacritics = '͜͡'
pre_diacritics, post_diacritics = [], []
for i in range(len(diacritics_data)):
    if diacritics_data['Position'][i] == 'pre':
        pre_diacritics.append(diacritics_data['Diacritic'][i])
    elif diacritics_data['Position'][i] == 'post':
        post_diacritics.append(diacritics_data['Diacritic'][i])
pre_diacritics = ''.join(pre_diacritics)
post_diacritics = ''.join(post_diacritics)
prepost_diacritics = {'ʰ', 'ʱ', 'ⁿ'} # diacritics which can appear before or after

# List of all diacritic characters
diacritics = ''.join([pre_diacritics, post_diacritics, inter_diacritics])


def strip_diacritics(string, excepted=[]):
    """Removes diacritic characters from an IPA string
    By default removes all diacritics; in order to keep certain diacritics,
    these should be passed as a list to the "excepted" parameter"""
    try:
        to_remove = ''.join([d for d in diacritics if d not in excepted])
        return re.sub(f'[{to_remove}]', '', string)

    except RecursionError:
        with open('error.out', 'w') as f:
            f.write(f'Unable to parse phonetic characters in form: {string}')
        raise RecursionError(f'Error parsing phonetic characters: see {os.path.join(os.getcwd(), "error.out")}')

def normalize_ipa_ch(string):
    """Normalizes some commonly mistyped IPA characters"""

    # <g> instead of <ɡ>
    string = re.sub('g', 'ɡ', string)

    # Affricates for which there is a special ligature character
    string = re.sub('t͡s', 'ʦ', string)
    string = re.sub('d͡z', 'ʣ', string)
    string = re.sub('t͡ʃ', 'ʧ', string)
    string = re.sub('d͡ʒ', 'ʤ', string)
    string = re.sub('t͡ɕ', 'ʨ', string)
    string = re.sub('d͡ʑ', 'ʥ', string)

    # Accented characters instead of vowel + tone diacritic
    string = re.sub('á', 'á', string)
    string = re.sub('à', 'à', string)
    string = re.sub('â', 'â', string)
    string = re.sub('ā', 'ā', string)
    string = re.sub('é', 'é', string)
    string = re.sub('è', 'è', string)
    string = re.sub('ê', 'ê', string)
    string = re.sub('ē', 'ē', string)
    string = re.sub('í', 'í', string)
    string = re.sub('ì', 'ì', string)
    string = re.sub('î', 'î', string)
    string = re.sub('ī', 'ī', string)
    string = re.sub('ó', 'ó', string)
    string = re.sub('ò', 'ò', string)
    string = re.sub('ô', 'ô', string)
    string = re.sub('ō', 'ō', string)
    string = re.sub('ú', 'ú', string)
    string = re.sub('ù', 'ù', string)
    string = re.sub('û', 'û', string)
    string = re.sub('ū', 'ū', string)
    string = re.sub('ý', 'ý', string)
    string = re.sub('ŕ', 'ŕ', string)

    # Vowels with tilde as single character instead of vowel + tilde (nasal) diacritic
    string = re.sub('ã', 'ã', string)
    string = re.sub('ẽ', 'ẽ', string)
    string = re.sub('ĩ', 'ĩ', string)
    string = re.sub('õ', 'õ', string)
    string = re.sub('ũ', 'ũ', string)

    # Cyrillic and Greek characters that look like Latin/IPA characters
    # Cyrillic
    string = re.sub('а', 'a', string)
    string = re.sub('е', 'e', string)
    string = re.sub('і', 'i', string)
    string = re.sub('о', 'o', string)
    string = re.sub('я', 'ʁ', string)
    string = re.sub('з', 'ɜ', string)
    # Greek
    string = re.sub('ο', 'o', string)
    string = re.sub('ε', 'ɛ', string)
    string = re.sub('λ', 'ʎ', string)
    string = re.sub('δ', 'ð', string)

    # Other
    string = re.sub('∅', 'ø', string)
    string = re.sub('エ', 'ɪ', string)
    string = re.sub("'", 'ˈ', string)
    string = re.sub(':', 'ː', string)

    return string

valid_ipa_ch = ''.join([all_sounds, diacritics, ' ', '‿'])
def invalid_ch(string, valid_ch=valid_ipa_ch):
    """Returns set of unrecognized (non-IPA) characters in phonetic string"""
    return set(re.findall(fr'[^{valid_ch}]', string))

def verify_charset(string):
    """Verifies that all characters are valid IPA characters or diacritics, otherwise raises error"""
    unk_ch = invalid_ch(string)
    if len(unk_ch) > 0:
        unk_ch_str = '>, <'.join(unk_ch)
        raise ValueError(f'Invalid IPA character(s) <{unk_ch_str}> found in "{string}"!')

def is_ch(ch, l):
    try:
        if strip_diacritics(ch)[0] in l:
            return True
        else:
            return False
    except IndexError:
        return False

def is_vowel(ch):
    return is_ch(ch, vowels)

def is_consonant(ch):
    return is_ch(ch, consonants)

def is_glide(ch):
    return is_ch(ch, glides)

def is_diphthong(seg):
    if re.search(fr'([{vowels}]̯[{vowels}])|([{vowels}][{vowels}]̯)', seg):
        return True
    return False

# BASIC PHONE ANALYSIS: Methods for yielding feature dictionaries of phone segments
phone_ids = {} # Dictionary of phone feature dicts # TODO Change to be dictionary {str:Segment class} 

def phone_id(segment):
    """Returns a dictionary of phonetic feature values for the segment"""
    
    # Try to retrieve pre-calculated feature dictionary, if available
    if segment in phone_ids:
        return phone_ids[segment]
    
    # Verify that all characters in phone are valid IPA characters
    verify_charset(segment)

    # Generate an empty phone feature dictionary
    seg_dict = defaultdict(lambda:0)
    
    # Split segment into component parts, if relevant
    parts = re.split('͡|͜', segment)
    bases = []
    
    # Generate feature dictionary for each part and add to main feature dict
    for part in parts:
        if len(part.strip()) > 0:

            # Base of the segment is the non-diacritic portion
            base = strip_diacritics(part)
            if len(base) == 0:
                raise AssertionError(f'Error: invalid segment <{segment}>, no base IPA character found!')
            bases.append(base)

            # If the length of the base > 1, the segment is a diphthong (e.g. /e̯a/) or complex toneme (e.g. /˥˩/)
            # Filter out tonemes
            if (len(base) > 1) and (base[0] not in tonemes):
                return diphthong_features(part)
            
            # If the segment is a toneme, use the first component as its base
            elif base[0] in tonemes:
                return tonal_features(segment)

            # Otherwise, retrieve the base phone's features
            else:
                base_id = {feature:phone_features[base][feature] for feature in phone_features[base]}
                modifiers = set(ch for ch in segment if ch not in base)
                if len(modifiers) > 0:
                    part_id = apply_diacritics(base, base_id, modifiers)
                else:
                    part_id = base_id
                
                # Add to overall segment ID
                for feature in part_id:
                    # Value = 1 (+) overrides value = 0 (-,0)
                    seg_dict[feature] = max(seg_dict[feature], part_id[feature])
    
    # Ensure that affricates are +DELAYED RELEASE and -CONTINUANT
    if len(parts) > 1:
        if bases[0] in plosive:
            if bases[-1] in fricative:
                seg_dict['delayedRelease'] = 1 
                seg_dict['continuant'] = 0 
    
    # Add segment's feature dictionary to phone_ids; return the feature dictionary
    phone_ids[segment] = seg_dict

    return seg_dict 

def diphthong_features(diphthong):
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
        feature_id = phone_id(component)
        for feature in feature_id:
            diphth_dict[feature] += (weight * feature_id[feature])
    
    # Length feature should be either 0 or 1
    if diphth_dict['long'] > 0:
        diphth_dict['long'] = 1
        
    return diphth_dict

def apply_diacritics(base:str, base_features:set, diacritics:set):
    """Applies feature values of diacritics to base segments

    Args:
        base (str): base IPA segment
        base_features (set): feature dictionary of base segment
        diacritics (set): diacritics to apply
    """

    # Apply diacritic effects to feature dictionary
    for modifier in diacritics:
        for feature, value in diacritics_effects[modifier]:
            base_features[feature] = value
            
        if modifier == '̞': # lowered diacritic: turns fricatives into approximants
            if base[0] in fricative:
                base_features['approximant'] = 1
                base_features['consonantal'] = 0
                base_features['delayedRelease'] = 0
                base_features['sonorant'] = 1
        
        elif modifier == '̝': # raised diacritic
            # turn approximants/trills into fricativized approximants
            if base[0] in approximants+trills:
                base_features['delayedRelease'] = 1
                
            # turn fricatives into plosives
            elif base[0] in fricative:
                base_features['continuant'] = 0
                base_features['delayedRelease'] = 0
    
    return base_features


tone_levels = {'˩':1, '¹':1, 
               '˨':2, '²':2,
               '˧':3, '³':3,
               '˦':4, '⁴':4, 
               '˥':5, '⁵':5,
               '↓':0, '⁰':0}

def tonal_features(toneme):
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


def prosodic_environment_weight(segments, i):
    """Returns the relative prosodic environment weight of a segment within
    a word, based on List (2012)"""
    
    # Word-initial segments
    if i == 0:
        # Word-initial consonants: weight 7
        if strip_diacritics(segments[i])[0] in consonants:
            return 7
        
        # Word-initial vowels: weight 6
        else:
            return 6
    
    # Word-final segments
    elif i == len(segments)-1:
        stripped_segment = strip_diacritics(segments[i])[0]
        
        # Word-final consonants: weight 2
        if stripped_segment in consonants:
            return 2
        
        # Word-final vowels: weight 1
        elif stripped_segment in vowels:
            return 1
        
        # Word-final tonemes: weight 0
        else:
            return 0
    
    # Word-medial segments
    else:
        prev_segment, segment_i, next_segment = segments[i-1], segments[i], segments[i+1]
        prev_sonority, sonority_i, next_sonority = map(get_sonority, [prev_segment, 
                                                                      segment_i, 
                                                                      next_segment])
        
        # Sonority peak: weight 3
        if prev_sonority <= sonority_i >= next_sonority:
            return 3
        
        # Descending sonority: weight 4
        elif prev_sonority >= sonority_i >= next_sonority:
            return 4
        
        # Ascending sonority: weight 5
        else:
            return 5
        
        # TODO: what if the sonority is all the same? add tests to ensure that all of these values are correct
        # TODO: sonority of free-standing vowels (and consonants)?: would assume same as word-initial


class Segment:
    segments = {}

    def __init__(self, segment):
        self.segment = normalize_ipa_ch(segment)

        # Base segment: no diacritics; first element of diphthongs, affricates, or complex consonants
        self.base = strip_diacritics(self.segment)[0]

        # Get distinctive phonological feature dictionary
        self.features = phone_id(self.segment) # TODO change to class method
        
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


    def get_phone_class(self):
        if is_diphthong(self.segment):
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


    def get_manner(self):
        if self.phone_class in ('CONSONANT', 'GLIDE'):
            if self.base in affricate or re.search(fr'{plosive}{diacritics}*͡{fricative}{diacritics}*', self.segment):
                return 'AFFRICATE'
            if self.base in plosive:
                return 'PLOSIVE'
            elif self.base in nasals:
                return 'NASAL'
            elif self.base in fricative and '̞' in self.segment: # lowered diacritic turns fricatives into approximants
                return 'APPROXIMANT'
            elif re.search('[ɬɮ]', self.base):
                return 'LATERAL FRICATIVE'
            elif self.base in fricative:
                return 'FRICATIVE'
            elif self.base in trills and '̝' in self.segment:
                return 'FRICATIVE TRILL'
            elif self.base in trills:
                return 'TRILL'
            elif self.base in tap_flap:
                return 'TAP/FLAP'
            elif self.features['lateral'] == 1 and self.features['approximant'] == 1:
                return 'LATERAL APPROXIMANT'
            elif self.base in approximant:
                return 'APPROXIMANT'
            elif self.base in implosive:
                return 'IMPLOSIVE'
            elif self.base in clicks:
                return 'CLICK'
            else:
                raise ValueError(f'Could not determine manner of articulation for {self.segment}')
        else:
            return self.phone_class
        

    def get_poa(self):
        val_err = ValueError(f'Could not determine place of articulation for {self.segment}')
        if self.phone_class in ('CONSONANT', 'GLIDE'):
            if self.base in bilabial:
                return 'BILABIAL'
            elif self.features['labiodental'] == 1:
                return 'LABIODENTAL'
            elif re.search(r'[θðǀ̪]', self.segment):
                return 'DENTAL'
            elif self.base in alveolar:
                return 'ALVEOLAR'
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
            elif self.base in uvular:
                return 'PHARYNGEAL'
            elif self.base in epiglottal:
                return 'EPIGLOTTAL'
            elif self.base in glottal:
                return 'GLOTTAL'
            else:
                raise val_err
            
        elif self.phone_class == 'VOWEL':
            # Height / Openness
            if re.search(r'[iyɨʉɯu]', self.base):
                height = 'CLOSE'
            elif re.search(r'[ɪʏʊeøɘɵɤo]', self.base):
                height = 'CLOSE-MID'
            elif re.search(r'[əɚ]', self.base):
                height = 'MID'
            elif re.search(r'[ɛœɜɞɝʌɔæɐ]', self.base):
                height = 'OPEN-MID'
            elif re.search(r'[aɶɑɒ]', self.base):
                height = 'OPEN'
            else:
                raise val_err
            
            # Frontness / Backness
            if re.search(r'[iyɪʏeøɛœæaɶ]', self.base):
                frontness = 'FRONT'
            elif re.search(r'[ɨʉɘɵəɜɞɐ]', self.base):
                frontness = 'CENTRAL'
            elif re.search(r'[ɯuʊɤoʌɔɑɒ]', self.base):
                frontness = 'BACK'
            else:
                raise val_err

            return ' '.join([height, frontness])

        elif self.phone_class == 'DIPHTHONG': # TODO
            raise NotImplementedError


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

                # Use only the syllabic component for sonority calculation # TODO maybe there is a better method? see below
                syl_comp = re.search(fr'[{vowels}](?![{post_diacritics}]*)', self.segment).group()
                raise NotImplementedError # TODO finish this
                strip_sound = strip_diacritics(syl_comp)
                phone = phone_id(syl_comp)

                # TODO: use this method instead
                # Diphthong: calculate sonority as maximum sonority of component parts
                if strip_sound[0] in vowels:
                    diphthong_components = segment_ipa(sound, combine_diphthongs=False)
                    sonorities = [get_sonority(v) for v in diphthong_components]
                    return max(sonorities)

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
            raise ValueError(f'Error: the sonority of phone "{sound}" cannot be determined!')


    def __str__(self):
        """Print the segment and its class, manner, place of articulation, sonority, and/or other relevant features."""
        info = [
            f'/{self.segment}/',
            f'Class: {self.phone_class}'
        ]

        if self.phone_class in ('CONSONANT', 'VOWEL', 'DIPHTHONG'):
            info.extend([
                f'Place of Articulation: {self.poa}',
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
        


def phonEnvironment(segments, i):
    """Returns a string representing the phonological environment of a segment within a word
    # TODO add front/back vowel context
    """
    # Designate first non-diacritic component of segment as base
    segment_i = segments[i]
    base = strip_diacritics(segment_i)[0]

    # Tonemes
    if base in tonemes: 
        # TODO should remove all tonemes from word and reevaluate without them, so that final segments are considered final despite being "followed" by a toneme
        return 'T'

    # Word-initial segments (free-standing segments also considered word-initial)
    elif i == 0:
        if len(segments) > 1:
            next_segment = segments[i+1]
            sonority_i, next_sonority = map(get_sonority, [segment_i, next_segment])
        else:
            next_segment = None

        # Word-initial segments: #S
        if next_segment:
            if segment_i == next_segment:
                return '#SS'
            elif sonority_i == next_sonority:
                return '#S='
            elif sonority_i > next_sonority:
                return '#S>'
            else: # sonority_i < next_sonority:
                return '#S<'
        
        # Free-standing segments
        else:
            return '#S#'
    
    # Word-final segments: S#
    elif i == len(segments)-1:
        assert len(segments) > 1
        prev_segment = segments[i-1]
        if prev_segment == segment_i:
            return 'SS#'
        else:
            prev_sonority, sonority_i = map(get_sonority, [prev_segment, segment_i])
            
            if prev_sonority == sonority_i:
                return '=S#'

            elif prev_sonority < sonority_i:
                return '<S#'

            else: # prev_sonority > sonority_i
                return '>S#' 
    
    # Word-medial segments
    else:
        prev_segment, next_segment = segments[i-1], segments[i+1]
        prev_sonority, sonority_i, next_sonority = map(get_sonority, [prev_segment, 
                                                                      segment_i, 
                                                                      next_segment])
        
        # Sonority plateau: =S=
        if prev_segment == sonority_i == next_sonority:
            return '=S='
        
        # Sonority peak: <S>
        elif prev_sonority < sonority_i > next_sonority:
            return '<S>'
        
        # Sonority trench: >S< # TODO is this the best term?
        elif prev_sonority > sonority_i < next_sonority:
            return '>S<'
        
        # Descending sonority: >S>
        elif prev_sonority > sonority_i > next_sonority:
            return '>S>'
        
        # Ascending sonority: <S<
        elif prev_sonority < sonority_i < next_sonority:
            return '<S<'
        
        elif prev_sonority < sonority_i == next_sonority:
            if segment_i == next_segment:
                return '<SS'
            else:
                return '<S='
        
        elif prev_sonority > sonority_i == next_sonority:
            if segment_i == next_segment:
                return '>SS'
            else:
                return '>S='
        
        elif prev_sonority == sonority_i < next_sonority:
            if segment_i == prev_segment:
                return 'SS<'
            else:
                return '=S<'
        
        elif prev_sonority == sonority_i > next_sonority:
            if segment_i == prev_segment:
                return 'SS>'
            else:
                return '=S>'
        
        else:
            raise NotImplementedError(f'Unable to determine environment for segment {i} /{segments[i]}/ within /{"".join(segments)}/')

# WORD SEGMENTATION
pre_preaspiration = vowels.union(glides).union(set(post_diacritics)) # characters which can occur before preaspiration characters <ʰʱ>, to distinguish from post-aspiration
segment_regexes = [
    fr'(?<=[{pre_preaspiration}])[{pre_diacritics}]*[ʰʱ][{pre_diacritics}]*[{consonants}][{post_diacritics}]*',
    fr'(?<=^)[{pre_diacritics}]*[ʰʱ][{pre_diacritics}]*[{consonants}][{post_diacritics}]*',
    fr'[{pre_diacritics}]*[{all_sounds}][{post_diacritics}]*',
]
segment_regex = '(' + '|'.join(segment_regexes) + ')'
segment_regex = re.compile(segment_regex)
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
            preasp = re.search(rf'(?<=[{pre_preaspiration}])[ʰʱ]$', seg)
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
            if '̯' in seg and is_vowel(seg):
                if i > 0:
                    # First try to combine with preceding vowel
                    if is_vowel(updated_segments[-1]):
                        updated_segments[-1] += seg
                        i += 1

                    # If there is no suitable preceding vowel, try combining with following vowel instead
                    elif is_vowel(segments[i+1]):
                        updated_segments.append(seg+segments[i+1])
                        i += 2

                    # Else do nothing
                    else:
                        updated_segments.append(seg)
                        i += 1

                # Combine an initial non-syllabic vowel onto a following vowel
                else:
                    if is_vowel(segments[1]):
                        updated_segments.append(seg+segments[1])
                        i += 2
            else:
                updated_segments.append(seg)
                i += 1
        
        segments = updated_segments


    return segments

def remove_stress(ipa):
    """Removes stress annotation from an IPA string"""
    return re.sub('[ˈˌ]', '', ipa)


def common_features(segment_list, start_features=features):
    """Returns the features/values shared by all segments in the list"""
    features = list(start_features)[:]
    feature_values = defaultdict(lambda:[])
    for seg in segment_list:
        for feature in features:
            value = phone_id(seg)[feature]
            if value not in feature_values[feature]:
                feature_values[feature].append(value)
    common = [(feature, feature_values[feature][0]) for feature in feature_values if len(feature_values[feature]) == 1]
    return common

def different_features(seg1, seg2, return_list=False):
    diffs = []
    seg1_id = phone_id(seg1)
    seg2_id = phone_id(seg2)
    for feature in seg1_id:
        if seg2_id[feature] != seg1_id[feature]:
            diffs.append(feature)
    if return_list:
        return diffs
    else:
        if len(diffs) > 0:
            print(f'\t\t\t{seg1}\t\t{seg2}')
            for feature in diffs:
                print(f'{feature}\t\t{seg1_id[feature]}\t\t{seg2_id[feature]}')

def lookup_segments(features, values, segment_list=all_sounds):
    """Returns a list of segments whose feature values match the search criteria"""
    matches = []
    for segment in segment_list:
        match_tallies = 0
        for feature, value in zip(features, values):
            if phone_id(segment)[feature] == value:
                match_tallies += 1
        if match_tallies == len(features):
            matches.append(segment)
    return set(matches)



# SIMILARITY / DISTANCE MEASURES

# Cosine similarity: cosine() imported from scipy.spatial.distance

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


# Feature Geometry Weights
# Feature weight calculated as ln(n_distinctions) / (tier**2)
# where n_distinctions = (n_sisters+1) + (n_descendants)
feature_geometry = pd.read_csv(os.path.join(save_dir, 'Phones/feature_geometry.tsv'), sep='\t')
feature_geometry['Tier'] = feature_geometry['Path'].apply(lambda x: len(x.split(' | ')))
feature_geometry['Parent'] = feature_geometry['Path'].apply(lambda x: x.split(' | ')[-1])
feature_geometry['N_Sisters'] = feature_geometry['Parent'].apply(lambda x: feature_geometry['Parent'].to_list().count(x))
feature_geometry['N_Descendants'] = feature_geometry['Feature'].apply(lambda x: len([i for i in range(len(feature_geometry)) 
                                                                                     if x in feature_geometry['Path'].to_list()[i].split(' | ')]))
feature_geometry['N_Distinctions'] = (feature_geometry['N_Sisters'] + 1) + (feature_geometry['N_Descendants'])
weights = [math.log(row['N_Distinctions']) / (row['Tier']**2) for index, row in feature_geometry.iterrows()]
total_weights = sum(weights)
normalized_weights = [w/total_weights for w in weights]
feature_geometry['Weight'] = normalized_weights
feature_weights = {feature_geometry.Feature.to_list()[i]:feature_geometry.Weight.to_list()[i]
                   for i in range(len(feature_geometry))}


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
checked_phone_sims = {}
def phone_sim(phone1, phone2, similarity='weighted_dice', exclude_features=[]):
    """Returns the similarity of the features of the two phones according to
    the specified distance/similarity function;
    Features not to be included in the comparison should be passed as a list to
    the exclude_features parameter (by default no features excluded)"""
    
    # If the phone similarity has already been calculated for this pair, retrieve it
    reference = (phone1, phone2, similarity, tuple(exclude_features))
    if reference in checked_phone_sims:
        return checked_phone_sims[reference]
    
    # Get feature dictionaries for each phone
    phone_id1, phone_id2 = phone_id(phone1), phone_id(phone2)
    
    # Remove any specified features
    for feature in exclude_features:
        for phoneid in [phone_id1, phone_id2]:
            try:
                del phoneid[feature]
            except KeyError:
                pass

    # Calculate similarity of phone features according to specified measure
    measures = {'cosine':cosine,
                'dice':dice_sim,
                'hamming':hamming_distance,
                'jaccard':jaccard_sim,
                'weighted_cosine':cosine,
                'weighted_dice':weighted_dice,
                'weighted_hamming':weighted_hamming,
                'weighted_jaccard':weighted_jaccard}
    try:
        measure = measures[similarity]
    except KeyError:
        raise KeyError(f'Error: similarity measure "{similarity}" not recognized!')
    
    if similarity not in ['cosine', 'weighted_cosine']:
        score = measure(phone_id1, phone_id2)
    else:
        compare_features = list(phone_id1.keys())
        phone1_values = [phone_id1[feature] for feature in compare_features]
        phone2_values = [phone_id2[feature] for feature in compare_features]
        if similarity == 'weighted_cosine':
            weights = [feature_weights[feature] for feature in compare_features]
            # Subtract from 1: cosine() returns a distance
            score = 1 - measure(phone1_values, phone2_values, w=weights)
        else:
            score = 1 - measure(phone1_values, phone2_values)
    
    # If method is Hamming, convert distance to similarity
    if similarity in ['hamming', 'weighted_hamming']:
        score = 1 - score
        
    # Save the phonetic similarity score to dictionary, return score
    checked_phone_sims[reference] = score
    checked_phone_sims[reference] = score
    return score

breakpoint()