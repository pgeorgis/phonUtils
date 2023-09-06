
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phonUtils.initPhoneData import diacritics, diacritic_regex, valid_ipa_ch, ipa_norm_map

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