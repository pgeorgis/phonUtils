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
    for ch in to_remove:
        string = re.sub(ch, '', string)
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
phone_groups = pd.read_csv(os.path.join(save_dir, 'Phones/phone_classes.csv'))
phone_groups = {phone_groups['Group'][i]:phone_groups['Phones'][i].split()
                for i in range(len(phone_groups))}

# Set these phone groups as global variables so that they are callable by name
globals().update(phone_groups)

# Set basic consonants and vowels using syllabic feature
consonants = set(phone for phone in phone_features
              if phone_features[phone]['syllabic'] == 0
              if phone not in tonemes)
vowels = set(phone for phone in phone_features if phone not in consonants.union(tonemes))

# All basic sounds
all_sounds = ''.join(consonants.union(vowels).union(tonemes))

# IMPORT DIACRITICS DATA
diacritics_data = pd.read_csv(os.path.join(save_dir, 'Phones', 'diacritics.csv'), sep='\t')

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


# BASIC PHONE ANALYSIS: Methods for yielding feature dictionaries of phone segments
phone_ids = {} # Dictionary of phone feature dicts 

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
    components = segment_ipa(diphthong)

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
        
        # Ensure that contour tones do not have features tone_mid and tone_central
        # (which are unique to level tones)
        toneme_id['tone_central'] = 0
        toneme_id['tone_mid'] = 0
    
        # Get the maximum tonal level
        max_level = max(toneme_levels)
        
        # Add feature tone_high if the maximum tone level is at least 4
        if max_level >= 4:
            toneme_id['tone_high'] = 1
        
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
    


def get_sonority(sound):
    """Returns the sonority level of a sound according to Parker's (2002) 
    universal sonority hierarchy
    
    modified:
    https://www.researchgate.net/publication/336652515/figure/fig1/AS:815405140561923@1571419143959/Adapted-version-of-Parkers-2002-sonority-hierarchy.ppm
    
    TO DO: DIPHTHONGS, Complex plosives, e.g. /k͡p̚/"""
    # If sonority for this sound has already been calculated, retrieve this
    if sound in phone_sonority:
        return phone_sonority[sound]
    
    # Strip diacritics
    strip_sound = strip_diacritics(sound)
    
    # Feature dictionary for sound
    phone = phone_id(sound)
    
    # Determine appropriate sonority level by checking membership in sound 
    # groups (manner/place of articulation) and/or relevant features     

    # Vowels
    if strip_sound in vowels:
        # Treat as glide if non-syllabic
        if phone['syllabic'] == 0:
            sonority = 11
        
        # Schwa /ə/ and /ɨ/ have special sonority
        elif strip_sound == 'ə':
            sonority = 13
        elif strip_sound == 'ɨ':
            sonority = 12
        
        # Open and near-open vowels
        elif ((phone['high'] == 0) and (phone['low'] == 1)): 
            sonority = 16
        
        # Open-mid, mid, close-mid vowels other than schwa /ə/
        elif phone['high'] == 0:  
            sonority = 15
        
        # Near-close and close vowels other than /ɨ/
        elif phone['high'] == 1:
            sonority = 14

    # Consonants
    elif strip_sound[0] in consonants: 
        # index 0, for affricates or complex plosives, such as /p͡f/ and /k͡p/, written with >1 character
        
        # Glides
        if strip_sound in glides:
            sonority = 11
        
        # /r/
        elif strip_sound == 'r':
            sonority = 10
        
        # Laterals
        elif phone['lateral'] == 1:
            sonority = 9
        
        # Taps/flaps
        elif strip_sound in tap_flap:
            sonority = 8
           
        # Trills
        elif strip_sound in trills:
            sonority = 7
        
        # Nasals
        elif strip_sound in nasals:
            sonority = 6
        
        # /h/
        elif strip_sound == 'h':
            sonority = 5
        
        # Fricatives
        elif strip_sound in fricative:
            
            # Voiced fricatives
            if phone['periodicGlottalSource'] == 1:
                sonority = 4
            
            # Voiceless fricatives
            else:
                sonority = 3
        
        # Affricates, plosives, implosives, clicks
        else:
        
            # Voiced
            if phone['periodicGlottalSource'] == 1:
                sonority = 2
                
            # Voiceless 
            else:
                sonority = 1
    
    # Tonemes
    elif strip_sound[0] in tonemes:
        sonority = 0
    
    # Other sounds: raise error message
    else:
        # Diphthong: calculate sonority as maximum sonority of component parts
        if strip_sound[0] in vowels:
            diphthong_components = segment_ipa(sound)
            sonorities = [get_sonority(v) for v in diphthong_components]
            sonority = max(sonorities)
            
        
        else:
            raise ValueError(f'Error: the sonority of phone "{sound}" cannot be determined!')
   
    # Save sonority level of this sound in sonority dictionary, return sonority level
    phone_sonority[sound] = sonority
    return sonority


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

# WORD SEGMENTATION
segment_regex = re.compile(f'[{pre_diacritics}]*[{all_sounds}][{post_diacritics}]*')
def segment_ipa(word, remove_ch=''):
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

    return segments

def remove_stress(ipa):
    """Removes stress annotation from an IPA string"""
    return re.sub('[ˈˌ]', '', ipa)


def common_features(segment_list, 
                    start_features=features):
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

def lookup_segments(features, values, 
                    segment_list=all_sounds):
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
feature_geometry = pd.read_csv(os.path.join(save_dir, 'Phones/feature_geometry.csv'), sep='\t')
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

def compare_measures(seg1, seg2):
    measures = {}
    for dist_func in ['cosine', 'hamming', 'jaccard', 'dice', 
                   'weighted_cosine', 'weighted_hamming', 'weighted_dice', 'weighted_jaccard']:
        measures[dist_func] = phone_sim(seg1, seg2, dist_func)
    return measures