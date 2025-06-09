import os
import re
from collections import defaultdict
from math import log

import pandas as pd

pre_diacritics, post_diacritics = [], []
suprasegmental_diacritics = set()
alveolopalatal, palatal, postalveolar = set(), set(), set()
affricates, approximants, fricatives, glides, liquids, nasals, plosives, tonemes = set(), set(), set(), set(), set(), set(), set(), set()
diacritics = ''

def binary_feature(feature):
    """Converts features of type ['0', '-', '+'] to binary [0, 1]"""
    if str(feature) == '+':
        return 1
    else:
        return 0


def load_phone_data(dir):
    phone_data = pd.read_csv(os.path.join(dir, 'phoneData', 'segments.tsv'), sep='\t')

    # Dictionary of basic phones with their phonetic features
    phone_features = {phone_data['segment'][i]:{feature:binary_feature(phone_data[feature][i])
                                            for feature in phone_data.columns
                                            if feature not in ['segment', 'sonority']
                                            if not pd.isnull(phone_data[feature][i])}
                    for i in range(len(phone_data))}

    features = set(feature for sound in phone_features for feature in phone_features[sound])

    # Load phone classes by manner and place of articulation, e.g. plosive, fricative, velar, palatal
    phone_classes = pd.read_csv(os.path.join(dir, 'phoneData/phone_classes.tsv'))
    phone_classes = {phone_classes['Group'][i]:set(phone_classes['Phones'][i].split())
                    for i in range(len(phone_classes))}
    
    return features, phone_features, phone_classes


def load_diacritics_data(dir):
    diacritics_data = pd.read_csv(os.path.join(dir, 'phoneData', 'diacritics.tsv'), sep='\t')

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
                                    if diacritics_data.Type[i] in ('suprasegmental', 'voice quality'))
    suprasegmental_diacritics.remove('ː') # don't include length as a suprasegmental
    suprasegmental_diacritics.remove('̆') # don't include length as a suprasegmental


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

    # Assemble dictionary of relevant data to return
    diacritics_data = {
        'diacritics':diacritics,
        'diacritics_effects':diacritics_effects,
        'suprasegmental_diacritics':suprasegmental_diacritics,
        'inter_diacritics':inter_diacritics,
        'pre_diacritics':pre_diacritics,
        'post_diacritics':post_diacritics,
        'prepost_diacritics':prepost_diacritics
    }

    return diacritics_data


def load_feature_geometry(dir):
    # Feature geometry weight calculated as ln(n_distinctions) / (tier**2)
    # where n_distinctions = (n_sisters+1) + (n_descendants)
    feature_geometry = pd.read_csv(os.path.join(dir, 'phoneData/feature_geometry.tsv'), sep='\t')
    feature_geometry['Tier'] = feature_geometry['Path'].apply(lambda x: len(x.split(' | ')))
    feature_geometry['Parent'] = feature_geometry['Path'].apply(lambda x: x.split(' | ')[-1])
    feature_geometry['N_Sisters'] = feature_geometry['Parent'].apply(lambda x: feature_geometry['Parent'].to_list().count(x))
    feature_geometry['N_Descendants'] = feature_geometry['Feature'].apply(lambda x: len([i for i in range(len(feature_geometry)) 
                                                                                        if x in feature_geometry['Path'].to_list()[i].split(' | ')]))
    feature_geometry['N_Distinctions'] = (feature_geometry['N_Sisters'] + 1) + (feature_geometry['N_Descendants'])
    weights = [log(row['N_Distinctions']) / (row['Tier']**2) for index, row in feature_geometry.iterrows()]
    total_weights = sum(weights)
    normalized_weights = [w/total_weights for w in weights]
    feature_geometry['Weight'] = normalized_weights
    feature_weights = {feature_geometry.Feature.to_list()[i]:feature_geometry.Weight.to_list()[i]
                    for i in range(len(feature_geometry))}
    
    return feature_weights


def load_ipa_norm_map(dir):
    map_file = os.path.join(dir, 'phoneData', 'ipa_normalization.map')
    ipa_norm_map = {}
    with open(map_file, 'r', encoding='utf-8') as map_f:
        for line in map_f.readlines():
            if not re.match(r'\s*#', line) and line.strip() != '':
                ch, repl = line.strip().split('\t')
                ipa_norm_map[ch] = repl

    return ipa_norm_map


def get_segmentation_regex(all_phones, consonants, pre_diacritics, post_diacritics, pre_preaspiration):
    consonants = ''.join(consonants)
    pre_preaspiration = ''.join(pre_preaspiration)
    segment_regexes = [
        fr'(?<=[{pre_preaspiration}])[{pre_diacritics}]*[ʰʱ][{pre_diacritics}]*[{consonants}][{post_diacritics}]*',
        fr'(?<=^)[{pre_diacritics}]*[ʰʱ][{pre_diacritics}]*[{consonants}][{post_diacritics}]*',
        fr'[{pre_diacritics}]*[{all_phones}][{post_diacritics}]*',
    ]
    segment_regex = '(' + '|'.join(segment_regexes) + ')'
    segment_regex = re.compile(segment_regex)

    return segment_regex


# INITIALIZE ALL CONSTANTS
dir = os.path.join(os.getcwd(), 'phyloLing', 'phonUtils')
features, phone_features, phone_classes = load_phone_data(dir)
diacritics_data = load_diacritics_data(dir)
ipa_norm_map = load_ipa_norm_map(dir)
feature_weights = load_feature_geometry(dir)

# Set contents of phone class and diacritics data dictionaries as global variables
globals().update(phone_classes)
globals().update(diacritics_data)

# Designate sets of consonants, vowels, all phones, and valid IPA characters 
consonants = set(phone for phone in phone_features
              if phone_features[phone]['syllabic'] == 0
              if phone not in tonemes)
vowels = set(phone for phone in phone_features if phone not in consonants.union(tonemes))
obstruents = set(plosives.union(affricates).union(fricatives))
sonorants = set(nasals.union(liquids).union(glides).union(approximants).union(set(vowels)))
all_phones = ''.join(consonants.union(vowels).union(tonemes))
valid_ipa_ch = ''.join([all_phones, diacritics, ' ', '‿', r'\-', r'\(', r'\)'])

# Create mapping of toneme letters to their relative pitch values # TODO could be loaded in from external file
tone_levels = {'˩':1, '¹':1, 
               '˨':2, '²':2,
               '˧':3, '³':3,
               '˦':4, '⁴':4, 
               '˥':5, '⁵':5,
               '↓':0, '⁰':0}

# IPA regex/constants
# Pre-preaspiration: characters which can occur before preaspiration characters <ʰʱ>, to distinguish from post-aspiration during segmentation
pre_preaspiration = vowels.union(glides).union(tonemes).union(suprasegmental_diacritics).union({'̯', 'ː', 'ˑ', '̆', '̃', '̟', '̠'})
preaspiration_regex = re.compile(rf'(?<=[{pre_preaspiration}])[ʰʱ]$')
segment_regex = get_segmentation_regex(all_phones, consonants, pre_diacritics, post_diacritics, pre_preaspiration)
vowel_str = ''.join(vowels)
plosive_str = ''.join(plosives)
fricative_str = ''.join(fricatives)
diacritic_str = ''.join(diacritics)
diacritic_regex = re.compile(rf'[{diacritic_str}]')
diphthong_regex = re.compile(fr'([{vowel_str}][{diacritic_str}]*̯[{diacritic_str}]*[{vowel_str}])|([{vowel_str}][{diacritic_str}]*[{vowel_str}][{diacritic_str}]*̯)')
affricate_regex = re.compile(rf'[{plosive_str}].*͡.*[{fricative_str}]')
geminate_regex = re.compile(rf'([{pre_diacritics}]*)([{consonants}])([{post_diacritics}]*)\1?\2\3?([{post_diacritics}]*)')
nasal_regex = re.compile(r'[̃mɱnɳɲŋɴ]')
rhotic_regex = re.compile(r'[rɹɺɻɽɾᴅʀʁɚɝ]|(.+˞)')
front_vowel_regex = re.compile(r'[iyɪʏeøɛœæaɶ]')
central_vowel_regex = re.compile(r'[ɨʉɘɵəɚɜɝɞɐ]')
back_vowel_regex = re.compile(r'[ɯuʊɤoʌɔɑɒ]')
close_vowel_regex = re.compile(r'[iyɨʉɯu]')
close_mid_vowel_regex = re.compile(r'[ɪʏʊeøɘɵɤo]')
open_mid_vowel_regex = re.compile(r'[ɛœɜɞɝʌɔæɐ]')
open_vowel_regex = re.compile(r'[aɶɑɒ]')
toneme_regex = re.compile(rf'[{tonemes}]')

# Auxiliary functions for certain phone classes
def _is_affricate(phone):
    if '͡' in phone:
        if affricate_regex.search(phone):
            return True
    elif phone in affricates:
        return True
    return False