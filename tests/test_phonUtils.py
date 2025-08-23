from constants import IPA_SEGMENTS
from ipaTools import strip_diacritics
from phonSim import phone_sim
from segment import segment_ipa

def test_strip_diacritics():
    string1 = 'dʲǐə̯vɐs'
    ref1 = 'diəvɐs'

    string2 = 'bérˀzʲas'
    ref2 = 'berzas'

    string3 = 'zʲwai̯zdáːˀ'
    ref3 = 'zwaizda'

    for string_i, ref_i in zip(
        [string1, string2, string3],
        [ref1, ref2, ref3],
    ):
        assert strip_diacritics(string_i) == ref_i
    
    # Test diacritics removal with exception of stress
    string4 = 'zˈuɔ̯b̥s'
    ref4a = 'zuɔbs'
    ref4b = 'zˈuɔbs'
    assert strip_diacritics(string4) == ref4a
    assert strip_diacritics(string4, excepted={'ˈ'}) == ref4b

def test_segmentation():
    for ipa, segments in [
        ('fɫˈoːrɛ̃ː', ['f', 'ɫ', 'ˈoː', 'r', 'ɛ̃ː']),
        ('ne˥wŋ͡m', ['n', 'e', '˥', 'w', 'ŋ͡m']),
        ('ðˤaːd͡ʒaχ', ['ðˤ', 'aː', 'd͡ʒ', 'a', 'χ']),
        ('ⁿdomˈates̠', ['ⁿd', 'o', 'm', 'ˈa', 't', 'e', 's̠']),
        ('fuːkʰalˈɔ̃jɕib̥ø̰', ['f', 'uː', 'kʰ', 'a', 'l', 'ˈɔ̃', 'j', 'ɕ', 'i', 'b̥', 'ø̰']),
        ('ˈɑːʰttɑ', ['ˈɑː', 'ʰt', 't', 'ɑ']),
    ]:
        assert segment_ipa(ipa) == segments


def test_autonomous_diacritic_segmentation():
    autonomous_diacritics = {'ˈ', 'ˀ', 'ˤ', '̌', '̀', '̂', '́'}
    
    string1 = 'dʲæ̌ːɕɪmt'
    ref1a = ['dʲ', 'æː', '̌', 'ɕ', 'ɪ', 'm', 't']
    ref1b = ['dʲ', 'æ̌ː', 'ɕ', 'ɪ', 'm', 't']
    
    string2 = 'dɔ́ːwɡ̊'
    ref2a = ['d', 'ɔː', '́', 'w', 'ɡ̊']
    ref2b = ['d', 'ɔ́ː', 'w', 'ɡ̊']

    string3 = 'ʥˈɛɕɛɲʨ'
    ref3a = ['ʥ', 'ɛ', 'ˈ', 'ɕ', 'ɛ', 'ɲ', 'ʨ']
    ref3b = ['ʥ', 'ˈɛ', 'ɕ', 'ɛ', 'ɲ', 'ʨ']

    string4 = 'ɡalˀwáːˀ'
    ref4a = ['ɡ', 'a', 'l', 'ˀ', 'w', 'aː', '́ˀ'] # TODO unsure if this is should be the default segmentation
    ref4b = ['ɡ', 'a', 'lˀ', 'w', 'áːˀ']

    string5 = 'ʧˤɑːv'
    ref5a = ['ʧ', 'ˤ', 'ɑː', 'v']
    ref5b = ['ʧˤ', 'ɑː', 'v']
    
    # Test with autonomous diacritic segmentation
    for string_i, ref_i in zip(
        [string1, string2, string3, string4, string5],
        [ref1a, ref2a, ref3a, ref4a, ref5a],
    ):
        assert segment_ipa(string_i, autonomous_diacritics=autonomous_diacritics) == ref_i
    # Test without autonomous diacritic segmentation
    for string_i, ref_i in zip(
        [string1, string2, string3, string4, string5],
        [ref1b, ref2b, ref3b, ref4b, ref5b],
    ):
        assert segment_ipa(string_i, autonomous_diacritics=None) == ref_i


def test_remove_ch_segmentation():
    string1 = ' lala '
    ref1 = ['l', 'a', 'l', 'a']
    
    string2 = 'lez‿ami'
    ref2 = ['l', 'e', 'z', 'a', 'm', 'i']
    
    # Test default remove_ch
    for string_i, ref_i in zip(
        [string1, string2],
        [ref1, ref2,],
    ):
        assert segment_ipa(string_i, remove_ch=None) == ref_i
        
    # Remove stress
    string3 = 'sulˈɛʎ'
    ref3 = ['s', 'u', 'l', 'ɛ', 'ʎ']
    assert segment_ipa(string3, remove_ch={'ˈ'}) == ref3


def test_combine_diphthongs_segmentation():
    string1 = 'drɐ̌ʊ̯ɡɐs'
    ref1a = ['d', 'r', 'ɐ̌ʊ̯', 'ɡ', 'ɐ', 's']
    ref1b = ['d', 'r', 'ɐ̌', 'ʊ̯', 'ɡ', 'ɐ', 's']
    
    string2 = 'zˈɛi̯də'
    ref2a = ['z', 'ˈɛi̯', 'd', 'ə']
    ref2b = ['z', 'ˈɛ', 'i̯', 'd', 'ə']
    
    string3 = 'ʃnˈaɪ̯'
    ref3a = ['ʃ', 'n', 'ˈaɪ̯']
    ref3b = ['ʃ', 'n', 'ˈa', 'ɪ̯']
    
    string4 = 'ou̯ɡɑ'
    ref4a = ['ou̯', 'ɡ', 'ɑ']
    ref4b = ['o', 'u̯', 'ɡ', 'ɑ']
 
    # Test with combine_diphthongs=True
    for string_i, ref_i in zip(
        [string1, string2, string3, string4],
        [ref1a, ref2a, ref3a, ref4a],
    ):
        assert segment_ipa(string_i, combine_diphthongs=True) == ref_i
    # Test with combine_diphthongs=False
    for string_i, ref_i in zip(
        [string1, string2, string3, string4],
        [ref1b, ref2b, ref3b, ref4b],
    ):
        assert segment_ipa(string_i, combine_diphthongs=False) == ref_i


def test_phone_sim_symmetrical():
    """Test that phone_sim(x, y) == phone_sim(y, x)."""
    for measure in [
        'cosine',
        'hamming',
        'jaccard',
        'weighted_dice',
        'weighted_hamming',
        'weighted_jaccard',
    ]:
        for x in IPA_SEGMENTS:
            for y in IPA_SEGMENTS:
                assert phone_sim(x, y, similarity=measure) == phone_sim(y, x, similarity=measure)


"""
PYTESTS TO ADD:
- Segment place of articulation, manner, voicing, tonal features
- Syllabification
- PhonEnv
"""