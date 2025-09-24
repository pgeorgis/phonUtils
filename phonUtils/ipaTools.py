# Functions for IPA string manipulation and normalization

import re

from phonUtils.constants import (DIACRITIC_REGEX, IPA_DIACRITICS, IPA_NORM_MAP,
                                 VALID_CHARACTERS)


def strip_diacritics(string: str, excepted: list = []):
    """Removes diacritic characters from an IPA string
    By default removes all diacritics; in order to keep certain diacritics,
    these should be passed as a list to the "excepted" parameter"""
    if len(excepted) > 0:
        to_remove = ''.join([d for d in IPA_DIACRITICS if d not in excepted])
        return re.sub(f'[{to_remove}]', '', string)
    else:
        return DIACRITIC_REGEX.sub('', string)


def normalize_ipa_ch(string: str, ipa_norm_map: dict = IPA_NORM_MAP) -> str:
    """Normalizes some commonly mistyped IPA characters according to a pre-loaded normalization mapping dictionary"""

    def replace_callback(match):
        return ipa_norm_map[match.group(0)]

    pattern = re.compile('|'.join(map(re.escape, ipa_norm_map.keys())))
    string = pattern.sub(replace_callback, string)

    return string


def invalid_ch(string: str, valid_ch: str = VALID_CHARACTERS) -> set:
    """Returns set of unrecognized (non-IPA) characters in phonetic string"""
    return set(re.findall(fr'[^{valid_ch}]', string))


def verify_charset(string: str) -> None:
    """Verifies that all characters are valid IPA characters or diacritics, otherwise raises error"""
    unk_ch = invalid_ch(string)
    if len(unk_ch) > 0:
        unk_ch_str = '>, <'.join(unk_ch)
        raise ValueError(f'Invalid IPA character(s) <{unk_ch_str}> found in "{string}"!')
