import pytest
import sys, os
sys.path.insert(0, '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1]))
from phonSim import segment_ipa

def test_segmentation():
    for ipa, segments in [
        ('fɫˈoːrɛ̃ː', ['f', 'ɫ', 'ˈoː', 'r', 'ɛ̃ː']),
        ('ne˥wŋ͡m', ['n', 'e', '˥', 'w', 'ŋ͡m']),
        ('ðˤaːd͡ʒaχ', ['ðˤ', 'aː', 'd͡ʒ', 'a', 'χ']),
        ('ⁿdomˈates̠', ['ⁿd', 'o', 'm', 'ˈa', 't', 'e', 's̠']),
        ('fuːkʰalˈɔ̃jɕib̥ø̰', ['f', 'uː', 'kʰ', 'a', 'l', 'ˈɔ̃', 'j', 'ɕ', 'i', 'b̥', 'ø̰']),
    ]:
        assert segment_ipa(ipa) == segments

"""
PYTESTS TO ADD:
- Segment place of articulation, manner, voicing, tonal features
"""