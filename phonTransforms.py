import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phonUtils import syllables
from phonUtils.initPhoneData import (consonants, fricatives, geminate_regex,
                                     plosives)
from phonUtils.segment import _toSegment, segment_ipa

VOICELESS_CONSONANTS: str = ''.join([phone for phone in consonants if _toSegment(phone).voiceless])
VOICED_CONSONANTS: str = ''.join([phone for phone in consonants if _toSegment(phone).voiced])

#General phonological transformation functions
devoice_dict = {
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

def finalDevoicing(word, phones, devoice_dict=devoice_dict):
    for phone in phones:
        devoiced = devoice_dict.get(phone, f'{phone}̥')
        word = re.sub(f'(?<!^){phone}(?![̥̊])(ʲ)?$', fr'{devoiced}\1', word)
    return word

def regressiveVoicingAssimilation(form,
                                  devoice_dict=devoice_dict,
                                  voicing_dict=None,
                                  to_voiceless=True,
                                  to_voiced=True,
                                  exception=[]):
    original = form[:]
    if voicing_dict is None:
        voicing_dict = {devoice_dict[p]:p for p in devoice_dict}
    voiced_str = '|'.join(devoice_dict.keys())
    voiceless_str = '|'.join(devoice_dict.values())

    # Voiced C1, voiceless C2
    if to_voiceless:
        for voiced, voiceless in devoice_dict.items():
            form = re.sub(rf'{voiced}(?![̥̊])(?=ʲ?([{VOICELESS_CONSONANTS}]|{voiceless_str}|.[̥̊]))', voiceless, form)

    # Voiceless C1, voiced C2
    if to_voiced:
        for voiceless, voiced in voicing_dict.items():
            form = re.sub(rf'{voiceless}(?=ʲ?(({voiced_str}|[{VOICED_CONSONANTS}])(?![̥̊])|.̬))', voiced, form)

    # Postprocess any misplaced diacritics
    form = re.sub(r'ʲ([̥̊])', r'\1ʲ', form)

    # Cancel the assimilation if it results in an illegal sequence
    for exc in exception:
        if re.search(exc, form):
            if not re.search(exc, original):
                return original

    return form

def degeminate(word, phones):
    for phone in phones:
        word = re.sub(f'{phone}{phone}', phone, word)
        word = re.sub(f'{phone}ː', phone, word)
    return word

def normalize_geminates(word: str) -> str:
    return geminate_regex.sub(r'\1\2\3\4ː', word)

def split_affricates(word) -> tuple[str, dict]:
    affricate_map = {
        'ʦ':'ts',
        'ʣ':'dz',
        'ʧ':'tʃ',
        'ʤ':'dʒ',
        'ʨ':'tɕ',
        'ʥ':'dʑ',
    }
    matched = {}
    for match in re.findall(rf'([{plosives}][͜͡][{fricatives}])', word):
        split_affr = re.sub('[͜͡]', '', match)
        word = re.sub(match, split_affr, word)
        matched[match] = split_affr

    for ligature, digraph in affricate_map.items():
        if ligature in word:
            word = re.sub(ligature, digraph, word)
            matched[ligature] = digraph

    return word, matched

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
    nostress = re.sub('[ˈˌ]','', word)
    syls = syllables.syllabify(nostress)
    syls = [syls[i].syl for i in syls]
    n_syl = min(n_syl, len(syls)-1)
    n_syl = max(n_syl, -len(syls))
    target_syl = segment_ipa(syls[n_syl])
    try:
        syllabic_i = syllables.findSyllabic(target_syl)[0]
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

def unstressedVowelReduction(word, vowels, reduced='ə'):
    # vowels as dict, reduced as None
    if reduced is None:
        if isinstance(vowels, dict):
            reduced_dict = vowels
        else:
            raise TypeError

    # reduced as str
    elif isinstance(reduced, str):
        if isinstance(vowels, str):
            reduced_dict = {vowels:reduced}
        elif isinstance(vowels, (list, tuple, set)):
            reduced = [reduced]*len(vowels)
            reduced_dict = {vowel:reduced_vowel for vowel, reduced_vowel in zip(vowels, reduced)}
        else:
            raise TypeError

    # reduced as iterable
    elif isinstance(reduced, (list, tuple, set)):
        assert len(vowels) == len(reduced)
        reduced_dict = {vowel:reduced_vowel for vowel, reduced_vowel in zip(vowels, reduced)}

    else:
        raise TypeError

    for vowel, reduced_vowel in reduced_dict.items():
        word = re.sub(fr'(?<![ˈˌ]){vowel}', reduced_vowel, word)

    return word
