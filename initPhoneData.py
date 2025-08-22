import os
import re
from collections import defaultdict
from math import log

import numpy as np
import pandas as pd

from constants import FILE_READER_DEFAULTS


def binary_feature(feature: str) -> int:
    """Converts features of type ['0', '-', '+'] to binary [0, 1]"""
    if feature == '+':
        return 1
    else:
        return 0


def load_phone_features(dir: str) -> tuple[set, dict]:
    """Loads phonological distinctive feature set and their values per IPA segment."""
    # Load segment features CSV
    phone_data = pd.read_csv(
        os.path.join(dir, 'phoneData', 'segments.tsv'),
        sep='\t', **FILE_READER_DEFAULTS
    )

    # Nested dictionary of basic phones with their phonetic features
    phone_features = {
        phone_data['segment'][i]: {
            feature: binary_feature(phone_data[feature][i]) 
            for feature in phone_data.columns
            if feature not in {'segment', 'sonority'}
            if not pd.isnull(phone_data[feature][i])
        }
        for i in range(len(phone_data))
    }

    # Get full feature set
    features = set(feature for sound in phone_features for feature in phone_features[sound])
    
    return features, phone_features


def load_phone_classes(dir: str) -> dict:
    """Load dictionary of phone classes."""
    # Load phone classes by manner and place of articulation, e.g. plosive, fricative, velar, palatal
    phone_classes = pd.read_csv(
        os.path.join(dir, 'phoneData', 'phone_classes.csv'),
        **FILE_READER_DEFAULTS
    )
    # Get sets of base characters per class
    phone_classes = {
        row['class']: set(row['phones'].split())
        for _, row in phone_classes.iterrows()
    }
    return phone_classes


def load_diacritics_data(dir: str) -> dict:
    # Load diacritics csv file
    diacritics_data = pd.read_csv(
        os.path.join(dir, 'phoneData', 'diacritics.tsv'),
        sep='\t',
        **FILE_READER_DEFAULTS
    )

    # Create dictionary of diacritic characters with affected features and values
    diacritics_effects = defaultdict(lambda:[])
    for _, row in diacritics_data.iterrows():
        effect = (row['feature'], binary_feature(row['value']))
        
        # Skip diacritics which have no effect on features
        if pd.notna(effect[0]):
            # Add to dictionary, with diacritic as key
            diacritics_effects[row['diacritic']].append(effect)

    # Isolate suprasegmental diacritics
    suprasegmental_diacritics = set(
        row['diacritic']
        for _, row in diacritics_data.iterrows()
        if row['type'] in ('suprasegmental', 'voice quality')
    )
    # Remove length diacritics from suprasegmentals
    suprasegmental_diacritics.remove('ː')
    suprasegmental_diacritics.remove('̆')

    # Diacritics by position with respect to base segments
    inter_diacritics = '͜͡'
    pre_diacritics, post_diacritics = [], []
    for _, row in diacritics_data.iterrows():
        if row['position'] == 'pre':
            pre_diacritics.append(row['diacritic'])
        elif row['position'] == 'post':
            post_diacritics.append(row['diacritic'])
    pre_diacritics = ''.join(pre_diacritics)
    post_diacritics = ''.join(post_diacritics)
    prepost_diacritics = {'ʰ', 'ʱ', 'ⁿ'} # diacritics which can appear before or after

    # String list of all diacritic characters
    diacritics = ''.join([pre_diacritics, post_diacritics, inter_diacritics])

    # Assemble dictionary of relevant data to return
    diacritics_data = {
        'diacritics': diacritics,
        'diacritics_effects': diacritics_effects,
        'suprasegmental_diacritics': suprasegmental_diacritics,
        'inter_diacritics': inter_diacritics,
        'pre_diacritics': pre_diacritics,
        'post_diacritics': post_diacritics,
        'prepost_diacritics': prepost_diacritics
    }

    return diacritics_data


def load_feature_geometry(dir: str) -> dict:
    """Load feature geometry and compute feature weights."""
    # Load feature geometry data from CSV
    feature_geometry = pd.read_csv(
        os.path.join(dir, 'phoneData', 'feature_geometry.tsv'),
        sep='\t',
        **FILE_READER_DEFAULTS
    )
    # Preprocess feature geometry data
    feature_geometry['tier'] = feature_geometry['path'].apply(lambda x: len(x.split(' | ')))
    feature_geometry['parent'] = feature_geometry['path'].apply(lambda x: x.split(' | ')[-1])
    feature_geometry['n_sisters'] = feature_geometry['parent'].apply(
        lambda x: feature_geometry['parent'].to_list().count(x)
    )
    feature_geometry['n_descendants'] = feature_geometry['feature'].apply(
        lambda x: len([
            i for i, row in feature_geometry.iterrows()
            if x in row['path'].split(' | ')
        ])
    )

    # Feature geometry weight calculated as ln(n_distinctions) / (tier**2)
    # where n_distinctions = (n_sisters+1) + (n_descendants)
    feature_geometry['n_distinctions'] = (feature_geometry['n_sisters'] + 1) + (feature_geometry['n_descendants'])
    weights = np.array([
        log(row['n_distinctions']) / (row['tier']**2)
        for _, row in feature_geometry.iterrows()
    ])
    total_weights = np.sum(weights)
    normalized_weights = weights / total_weights
    feature_geometry['weight'] = normalized_weights
    feature_weights = {
        row['feature']: row['weight']
        for _, row in feature_geometry.iterrows()
    }

    return feature_weights


def load_ipa_norm_map(dir: str) -> dict:
    """Load IPA normalization map."""
    map_file = os.path.join(dir, 'phoneData', 'ipa_normalization.map')
    ipa_norm_map = {}
    with open(map_file, 'r', **FILE_READER_DEFAULTS) as map_f:
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
script_directory = os.path.dirname(__file__)
features, phone_features = load_phone_features(script_directory)
phone_classes = load_phone_classes(script_directory)
diacritics_data = load_diacritics_data(script_directory)
ipa_norm_map = load_ipa_norm_map(script_directory)
feature_weights = load_feature_geometry(script_directory)

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
nasal_regex = re.compile(r'[̃mɱnɳɲŋɴᵐᶬⁿᵑ]')
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
