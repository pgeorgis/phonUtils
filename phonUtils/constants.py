import os
import re
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

FILE_READER_DEFAULTS = {
    'encoding': 'utf-8',
}

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


def get_segmentation_regex(all_phones: str,
                           consonants: str,
                           pre_diacritics: str,
                           post_diacritics: str,
                           pre_preaspiration: str):
    segment_regexes = [
        fr'(?<=[{pre_preaspiration}])[{pre_diacritics}]*[ʰʱ][{pre_diacritics}]*[{consonants}][{post_diacritics}]*',
        fr'(?<=^)[{pre_diacritics}]*[ʰʱ][{pre_diacritics}]*[{consonants}][{post_diacritics}]*',
        fr'[{pre_diacritics}]*[{all_phones}][{post_diacritics}]*',
    ]
    segment_regex = '(' + '|'.join(segment_regexes) + ')'
    segment_regex = re.compile(segment_regex)

    return segment_regex


# INITIALIZE ALL CONSTANTS
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
FEATURE_SET, PHONE_FEATURES = load_phone_features(ROOT_DIR)
PHONE_CLASSES = load_phone_classes(ROOT_DIR)
DIACRITICS_DATA = load_diacritics_data(ROOT_DIR)
IPA_NORM_MAP = load_ipa_norm_map(ROOT_DIR)
FEATURE_WEIGHTS = load_feature_geometry(ROOT_DIR)

# Set phone classes as constants
# Place of articulation
BILABIALS: set = PHONE_CLASSES['bilabial']
LABIODENTALS: set = PHONE_CLASSES['labiodental']
DENTALS: set = PHONE_CLASSES['dental']
ALVEOLARS: set = PHONE_CLASSES['alveolar']
POSTALVEOLARS: set = PHONE_CLASSES['postalveolar']
ALVEOLOPALATALS: set = PHONE_CLASSES['alveolopalatal']
RETROFLEXES: set = PHONE_CLASSES['retroflex']
PALATALS: set = PHONE_CLASSES['palatal']
VELARS: set = PHONE_CLASSES['velar']
UVULARS: set = PHONE_CLASSES['uvular']
PHARYNGEALS: set = PHONE_CLASSES['pharyngeal']
EPIGLOTTALS: set = PHONE_CLASSES['epiglottal']
GLOTTALS: set = PHONE_CLASSES['glottal']
LATERALS: set = PHONE_CLASSES['lateral']
# Manner of articulation
PLOSIVES: set = PHONE_CLASSES['plosives']
IMPLOSIVES: set = PHONE_CLASSES['implosives']
AFFRICATES: set = PHONE_CLASSES['affricates']
FRICATIVES: set = PHONE_CLASSES['fricatives']
NASALS: set = PHONE_CLASSES['nasals']
TRILLS: set = PHONE_CLASSES['trills']
TAPS_AND_FLAPS: set = PHONE_CLASSES['taps_flaps']
LIQUIDS: set = PHONE_CLASSES['liquids']
RHOTICS: set = PHONE_CLASSES['rhotics']
APPROXIMANTS: set = PHONE_CLASSES['approximants']
GLIDES: set = PHONE_CLASSES['glides']
CLICKS: set = PHONE_CLASSES['clicks']
TONEMES: set = PHONE_CLASSES['tonemes']
# Set IPA diacritic constants
IPA_DIACRITICS: str = DIACRITICS_DATA['diacritics']
PRE_DIACRITICS: str = DIACRITICS_DATA['pre_diacritics']
POST_DIACRITICS: str = DIACRITICS_DATA['post_diacritics']
SUPRASEGMENTAL_DIACRITICS: set = DIACRITICS_DATA['suprasegmental_diacritics']
DIACRITICS_EFFECTS: defaultdict = DIACRITICS_DATA['diacritics_effects']

# Designate sets of consonants, vowels, all phones, and valid IPA characters
CONSONANTS = set(
    phone for phone in PHONE_FEATURES
    if PHONE_FEATURES[phone]['syllabic'] == 0
    if phone not in TONEMES
)
VOWELS = set(
    phone for phone in PHONE_FEATURES
    if phone not in CONSONANTS.union(TONEMES)
)
OBSTRUENTS = PLOSIVES.union(AFFRICATES).union(FRICATIVES)
SONORANTS = NASALS.union(LIQUIDS).union(GLIDES).union(APPROXIMANTS).union(VOWELS)
IPA_SEGMENTS = CONSONANTS.union(VOWELS).union(TONEMES)
IPA_CHARACTERS = IPA_SEGMENTS.union(IPA_DIACRITICS)
SEGMENTATION_CH = {
        ' ',  # white space
        '‿',  # liaison tie
        r'\.',  # syllable boundary
        r'\-',  # morphological boundary
        r'\(',  # optional unit start
        r'\)',  # optional unit end
}
VALID_CHARACTERS = ''.join(IPA_CHARACTERS.union(SEGMENTATION_CH))

# Create mapping of toneme letters to their relative pitch values
TONE_LEVELS = {
    '˩': 1, '¹': 1,
    '˨': 2, '²': 2,
    '˧': 3, '³': 3,
    '˦': 4, '⁴': 4,
    '˥': 5, '⁵': 5,
    '↓': 0, '⁰': 0,
}
# Create mapping of tone diacritics to tone characters
TONE_DIACRITICS_MAP = {
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

# IPA regexes
# "Pre-preaspiration": characters which can occur before preaspiration characters <ʰʱ>, to distinguish from post-aspiration during segmentation
PRE_PREASPIRATION_CH = VOWELS.union(GLIDES).union(TONEMES).union(SUPRASEGMENTAL_DIACRITICS).union({'̯', 'ː', 'ˑ', '̆', '̃', '̟', '̠'})
PREASPIRATION_REGEX = re.compile(rf'(?<=[{PRE_PREASPIRATION_CH}])[ʰʱ]$')
SEGMENT_REGEX = get_segmentation_regex(
    ''.join(IPA_SEGMENTS),
    ''.join(CONSONANTS),
    PRE_DIACRITICS,
    POST_DIACRITICS,
    ''.join(PRE_PREASPIRATION_CH)
)
DIACRITIC_REGEX = re.compile(rf'[{IPA_DIACRITICS}]')
DIPHTHONG_REGEX = re.compile(fr'([{VOWELS}][{IPA_DIACRITICS}]*̯[{IPA_DIACRITICS}]*[{VOWELS}])|([{VOWELS}][{IPA_DIACRITICS}]*[{VOWELS}][{IPA_DIACRITICS}]*̯)')
AFFRICATE_REGEX = re.compile(rf'[{PLOSIVES}].*͡.*[{FRICATIVES}]')
GEMINATE_REGEX = re.compile(rf'([{PRE_DIACRITICS}]*)([{CONSONANTS}])([{POST_DIACRITICS}]*)\1?\2\3?([{POST_DIACRITICS}]*)')
NASAL_REGEX = re.compile(r'[̃mɱnɳɲŋɴᵐᶬⁿᵑ]')
RHOTIC_REGEX = re.compile(r'[rɹɺɻɽɾᴅʀʁɚɝ]|(.+˞)')
FRONT_VOWEL_REGEX = re.compile(r'[iyɪʏeøɛœæaɶ]')
CENTRAL_VOWEL_REGEX = re.compile(r'[ɨʉɘɵəɚɜɝɞɐ]')
BACK_VOWEL_REGEX = re.compile(r'[ɯuʊɤoʌɔɑɒ]')
CLOSE_VOWEL_REGEX = re.compile(r'[iyɨʉɯu]')
CLOSE_MID_VOWEL_REGEX = re.compile(r'[ɪʏʊeøɘɵɤo]')
OPEN_MID_VOWEL_REGEX = re.compile(r'[ɛœɜɞɝʌɔæɐ]')
OPEN_VOWEL_REGEX = re.compile(r'[aɶɑɒ]')
TONEME_REGEX = re.compile(rf'[{TONEMES}]')
