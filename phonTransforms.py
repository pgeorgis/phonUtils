import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phonUtils.initPhoneData import plosives, fricatives
from phonUtils.segment import segment_ipa
from phonUtils import syllables

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
        word = re.sub(f'{phone}(ʲ)?$', fr'{devoiced}\1', word)
    return word


def degeminate(word, phones):
    for phone in phones:
        word = re.sub(f'{phone}{phone}', phone, word)
        word = re.sub(f'{phone}ː', phone, word)
    return word


def split_affricates(word):
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

    nostress = re.sub('[ˈˌ]','', word)
    syls = syllables.syllabify(nostress)
    syls = [syls[i].syl for i in syls]
    n_syl = min(n_syl, len(syls)-1)
    n_syl = max(n_syl, -len(syls))
    target_syl = segment_ipa(syls[n_syl])
    syllabic_i = syllables.findSyllabic(target_syl)[0]
    target_syl.insert(syllabic_i, ch)
    target_syl = ''.join(target_syl)
    syls[n_syl] = target_syl
    
    return ''.join(syls)
