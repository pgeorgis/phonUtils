import re
from typing import Iterable

from . import syllables
from .constants import (CONSONANTS, FRICATIVES, GEMINATE_REGEX, PLOSIVES,
                        POST_DIACRITICS, VOWELS)
from .segment import Segment, segment_ipa

VOICELESS_CONSONANTS = ''.join([phone for phone in CONSONANTS if Segment(phone).voiceless])
VOICED_CONSONANTS = ''.join([phone for phone in CONSONANTS if Segment(phone).voiced])


DEFAULT_DEVOICE_DICT = {
    'b':'p',
    'd':'t',
    'ɟ':'c',
    'ɡ':'k',
    'ʣ':'ʦ',
    'ʤ':'ʧ',
    'ʥ':'ʨ',
    'v':'f',
    'ð':'θ',
    'z':'s',
    'ʒ':'ʃ',
    'ʐ':'ʂ',
    'ɣ':'x',
    'ʁ':'χ',
    'ɦ':'h',
    }


def finalDevoicing(ipa_string: str,
                   phones: Iterable = None,
                   devoice_dict: dict = DEFAULT_DEVOICE_DICT,
                   ):
    if phones is None:
        phones = devoice_dict.keys()
    for phone in phones:
        devoiced = devoice_dict.get(phone, f'{phone}̥')
        ipa_string = re.sub(f'(?<!^){phone}(?![̥̊])(ʲ)?$', fr'{devoiced}\1', ipa_string)
    return ipa_string


def regressiveVoicingAssimilation(ipa_string: str,
                                  devoice_dict: dict = DEFAULT_DEVOICE_DICT,
                                  voicing_dict: dict = None,
                                  to_voiceless: bool = True,
                                  to_voiced: bool = True,
                                  exception: Iterable = [],
                                  ):
    original = ipa_string[:]
    if voicing_dict is None:
        voicing_dict = {devoice_dict[p]:p for p in devoice_dict}
    voiced_str = '|'.join(devoice_dict.keys())
    voiceless_str = '|'.join(devoice_dict.values())

    # Voiced C1, voiceless C2
    if to_voiceless:
        for voiced, voiceless in devoice_dict.items():
            ipa_string = re.sub(rf'{voiced}(?![̥̊])(?=ʲ?([{VOICELESS_CONSONANTS}]|{voiceless_str}|.[̥̊]))', voiceless, ipa_string)

    # Voiceless C1, voiced C2
    if to_voiced:
        for voiceless, voiced in voicing_dict.items():
            ipa_string = re.sub(rf'{voiceless}(?=ʲ?(({voiced_str}|[{VOICED_CONSONANTS}])(?![̥̊])|.̬))', voiced, ipa_string)

    # Postprocess any misplaced diacritics
    ipa_string = re.sub(r'ʲ([̥̊])', r'\1ʲ', ipa_string)

    # Cancel the assimilation if it results in an illegal sequence
    for exc in exception:
        if re.search(exc, ipa_string):
            if not re.search(exc, original):
                return original

    return ipa_string


def degeminate(ipa_string: str,
               phones: Iterable = CONSONANTS,
               ):
    for phone in phones:
        ipa_string = re.sub(fr'({phone}){phone}(?!̩)([{POST_DIACRITICS}]*)',  r'\1\2', ipa_string)
        ipa_string = re.sub(fr'({phone})ː([{POST_DIACRITICS}]*)', r'\1\2', ipa_string)
        ipa_string = re.sub(fr'({phone})([{POST_DIACRITICS}]*)ː', r'\1\2', ipa_string)
    return ipa_string


def normalize_geminates(ipa_string: str):
    return GEMINATE_REGEX.sub(r'\1\2\3\4ː', ipa_string)


def split_affricates(ipa_string: str):
    affricate_map = {
        'ʦ':'ts',
        'ʣ':'dz',
        'ʧ':'tʃ',
        'ʤ':'dʒ',
        'ʨ':'tɕ',
        'ʥ':'dʑ',
    }
    matched = {}
    for match in re.findall(rf'([{PLOSIVES}][͜͡][{FRICATIVES}])', ipa_string):
        split_affr = re.sub('[͜͡]', '', match)
        ipa_string = re.sub(match, split_affr, ipa_string)
        matched[match] = split_affr

    for ligature, digraph in affricate_map.items():
        if ligature in ipa_string:
            ipa_string = re.sub(ligature, digraph, ipa_string)
            matched[ligature] = digraph

    return ipa_string, matched


def shiftStress(word, n_syl, type='PRIMARY'):
    """Shifts or adds stress to the nth syllable"""

    if type == 'PRIMARY':
        ch = 'ˈ'
    elif type == 'SECONDARY':
        ch = 'ˌ'
    else:
        raise ValueError(f'Error: unrecognized type "{type}". Must be one of "PRIMARY", "SECONDARY"')
    return shiftAccent(word, n_syl, ch)


def shiftAccent(word, n_syl, accent_ch='ˈ'):
    """Shifts or adds accent (pitch accent or stress) to the nth syllable"""
    no_accent = re.sub(accent_ch, '', word)
    syls = syllables.syllabify(no_accent)
    syls = [syls[i].syl for i in syls]
    n_syl = min(n_syl, len(syls)-1)
    n_syl = max(n_syl, -len(syls))
    target_syl = segment_ipa(syls[n_syl])
    try:
        syllabic_i = syllables.findSyllabicIndices(target_syl)[0]
        if accent_ch in {'ˈ', 'ˌ'}:
            target_syl.insert(syllabic_i, accent_ch)
        else:
            target_syl.insert(syllabic_i + 1, accent_ch)
        target_syl = ''.join(target_syl)
        target_syl = re.sub(fr'([ːˑ])({accent_ch})', r'\2\1', target_syl)
        syls[n_syl] = target_syl
    except IndexError:
        # No syllabic segment found in target syllable, skip
        pass

    return ''.join(syls)


def unstressedVowelReduction(ipa_string,
                             vowels: str | Iterable = VOWELS,
                             reduced: str | Iterable = 'ə',
                             reduction_dict: dict = None,
                             reduce_diphthongs: bool = True,
                             ):
    # use specified reduction dict
    if reduction_dict is not None:
        pass

    # vowels as dict, reduced as None
    elif reduced is None:
        if isinstance(vowels, dict):
            reduction_dict = vowels
        else:
            raise TypeError

    # reduced as str
    elif isinstance(reduced, str):
        if isinstance(vowels, str):
            reduction_dict = {vowels:reduced}
        elif isinstance(vowels, Iterable):
            reduced = [reduced]*len(vowels)
            reduction_dict = {vowel:reduced_vowel for vowel, reduced_vowel in zip(vowels, reduced)}
        else:
            raise TypeError

    # reduced as iterable
    elif isinstance(reduced, Iterable):
        assert len(vowels) == len(reduced)
        reduction_dict = {vowel:reduced_vowel for vowel, reduced_vowel in zip(vowels, reduced)}

    else:
        raise TypeError

    for vowel, reduced_vowel in reduction_dict.items():
        if reduce_diphthongs:
            ipa_string = re.sub(fr'(?<![ˈˌ]){vowel}', reduced_vowel, ipa_string)
        else:
            ipa_string = re.sub(fr'(?<![ˈˌ]){vowel}(?![{POST_DIACRITICS}]*̯)', reduced_vowel, ipa_string)

    return ipa_string
