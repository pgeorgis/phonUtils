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