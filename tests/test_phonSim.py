from constants import IPA_SEGMENTS
from phonSim import phone_sim
from segment import segment_ipa


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
    ref4a = ['ɡ', 'a', 'l', 'ˀ', 'w', 'aː', '́ˀ']
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