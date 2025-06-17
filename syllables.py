import os
import random
import re
import sys

from more_itertools import consecutive_groups

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phonUtils import phonTransforms
from phonUtils.initPhoneData import affricates, vowels
from phonUtils.segment import _is_vowel, _toSegment, segment_ipa


# Functions related to syllable types
def isSyllabic(segment):
    segment = _toSegment(segment)
    if segment.features['syllabic'] > 0:
        return True
    else:
        return False


def findSyllabic(segments):
    """Returns indices of syllabic segments"""
    return [i for i in range(len(segments)) if isSyllabic(segments[i])]


def countMorae(form, codaMora=False, **kwargs):
    """Returns the number of morae in an IPA string
    codaMora: if True, the length of the syllable coda is counted toward the number of morae"""
    syls = syllabify(form, **kwargs)
    morae = 0
    vowel_str = ''.join(vowels)
    for syl in syls:
        if syls[syl].nucleus:
            nucleus = syls[syl].nucleus[0]
            morae += len(re.findall(fr'[{vowel_str}ˑː̩]', nucleus))
            if codaMora:
                coda = syls[syl].coda
                morae += len(coda)
    return morae


def splitSyl(syl):
    """Splits a syllable into onset, nucleus, coda"""
    nucleus = findSyllabic(syl)
    n_syl = len(nucleus)
    try:
        assert n_syl == 1

        onset = syl[:nucleus[0]]
        try:
            coda = syl[nucleus[0]+1:]
        except IndexError:
            coda = []
        nucleus = [syl[nucleus[0]]]

        return onset, nucleus, coda

    except AssertionError:
        syl_str = ''.join(syl)
        if n_syl > 1:
            raise AssertionError(f'Error: "{syl_str}" contains >1 ({n_syl}) syllabic units!')
        else:
            raise AssertionError(f'Error: "{syl_str}" contains zero syllabic units!')


def sylType(syl, g_open=True):
    """Returns either OPEN or CLOSED, according to whether the input segments constitute an open or closed syllable
    g_open :: whether to consider syllables ending in glides open (default = True)"""
    n_syl = len(findSyllabic(syl))
    try:
        assert n_syl == 1

        finalSeg = syl[-1]

        if _is_vowel(finalSeg):
            return 'OPEN'

        else:
            if g_open:
                finalSeg = _toSegment(finalSeg)
                if finalSeg.phone_class == 'GLIDE':
                    return 'OPEN'
                else:
                    return 'CLOSED'
            else:
                return 'CLOSED'

    except AssertionError:
        syl_str = ''.join(syl)
        if n_syl > 1:
            raise AssertionError(f'Error: "{syl_str}" contains >1 ({n_syl}) syllabic units!')
        else:
            raise AssertionError(f'Error: "{syl_str}" contains zero syllabic units!')


class Syllable:
    def __init__(self, syl, g_open=True):
        self.g_open = g_open
        if type(syl) == str:
            self.syl = syl
            self.segments = segment_ipa(syl)
        elif type(syl) == list:
            self.syl = ''.join(syl)
            self.segments = syl
        else:
            raise ValueError
        if len(findSyllabic(syl)) > 0:
            self.onset, self.nucleus, self.coda = splitSyl(self.segments)
            self.type = sylType(syl, g_open=self.g_open)
        else: # No syllabic units found
            self.onset, self.nucleus, self.coda = self.segments, [], []
            self.type = "OTHER"
        self.stressed = 'STRESSED' if "ˈ" in self.syl else 'UNSTRESSED'

    def __str__(self):
        return self.syl


def syllabify(word,
              segments=None,
              illegal_coda=[],
              illegal_onset=[],
              default_coda=True,
              g_open=True,
              split_affricate=False,
              **kwargs,
              ):
    if segments is None:
        segments = segment_ipa(word)
    syllabic_i = findSyllabic(segments)

    # Split affricates across syllable boundaries (False by default)
    # Doesn't make any difference for monosyllabic words, so skip if <2 syllables
    if split_affricate and len(syllabic_i) > 1:
        word, matched_affricates = phonTransforms.split_affricates(word)
        segments = segment_ipa(word)
        syllabic_i = findSyllabic(segments)

    # try:
    #     assert len(syllabic_i) >= 1
    # except AssertionError:
    #     raise AssertionError(f'Error: no syllabic segments found in "{word}"')
    # If no syllabic units are found, return the entire word as a single syllable
    if len(syllabic_i) == 0:
        return {0:Syllable(segments, g_open=g_open)}

    syllables = {i:[segments[i]] for i in syllabic_i}
    onsets, codas = [], []
    for i in syllables:
        if i > 0:
            if i-1 not in syllables:
                if not any(re.search(Xonset, segments[i-1]) for Xonset in illegal_onset):
                    syllables[i].insert(0, segments[i-1])
                    onsets.append(i-1)

    # Automatically assign any non-syllabic units preceding the first syllabic nucleus to its onset, regardless of legality
    syl1_i = min(syllabic_i)
    if syl1_i > 0:
        if len(onsets) > 0:
            o = min(syl1_i, min(onsets))
        else:
            o = syl1_i
        for i in range(o-1,-1,-1):
            syllables[syl1_i].insert(0, segments[i])
            onsets.append(i)

    # Automatically assign any non-syllabic units following the last syllabic nucleus to its coda, regardless of legality
    sylN_i = max(syllabic_i)
    if sylN_i < len(segments)-1:
        for i in range(sylN_i+1, len(segments)):
            syllables[sylN_i].append(segments[i])
            codas.append(i)

    # Identify indices of segments which are not yet assigned as syllable nuclei, onsets, or codas
    unassigned = [j for j in range(len(segments)) if j not in syllabic_i if j not in onsets if j not in codas]

    # Combine unassigned indices into consecutive groups
    subsets = [list(group) for group in consecutive_groups(unassigned)]

    # Iterate through unassigned segment subsets
    for seq in subsets:
        start, end = seq[0], seq[-1]

        # Find the preceding and following syllables to sequence
        positions = sorted(seq + syllabic_i)
        start_index = positions.index(start)
        end_index = positions.index(end)
        preSyl = syllables[positions[start_index-1]]
        postSyl = syllables[positions[end_index+1]]

        preSyl_score = scoreSyl(
            preSyl,
            illegal_coda=illegal_coda,
            illegal_onset=illegal_onset,
            **kwargs
        )
        postSyl_score = scoreSyl(
            postSyl,
            illegal_coda=illegal_coda,
            illegal_onset=illegal_onset,
            **kwargs
        )
        start_score = preSyl_score + postSyl_score

        q_scores = {}
        q_syls = {}
        for q in range(len(seq)+1):
            q_coda, q_onset = [segments[s] for s in seq[:q]], [segments[s] for s in seq[q:]]
            hypPre = preSyl + q_coda
            hypPost = q_onset + postSyl
            hypPre_score = scoreSyl(
                hypPre,
                illegal_coda=illegal_coda,
                illegal_onset=illegal_onset,
                **kwargs
            )
            hypPost_score = scoreSyl(
                hypPost,
                illegal_coda=illegal_coda,
                illegal_onset=illegal_onset,
                **kwargs
            )
            q_scores[q] = (hypPre_score + hypPost_score) - start_score
            q_syls[q] = hypPre, hypPost
        min_q = min(q_scores.values())
        min_q_args = [q for q in q_scores if q_scores[q] == min_q]
        if len(min_q_args) > 1:
            q_syl = random.choice(min_q_args)
        else:
            q_syl = min_q_args[0]
        newPre, newPost = q_syls[q_syl]

        syllables[positions[start_index-1]] = newPre
        syllables[positions[end_index+1]] = newPost

    # Convert split affricates back
    if split_affricate and len(syllabic_i) > 1:
        syllable_indices = list(syllables.keys())
        for i in range(1, len(syllable_indices)):
            index1, index2 = syllable_indices[i-1], syllable_indices[i]
            syl1, syl2 = syllables[index1], syllables[index2]
            for affr, split_affr in matched_affricates.items():
                plosive_part, fricative_part = list(split_affr)
                if syl1[-1] == plosive_part and syl2[0] == fricative_part:
                    syllables[index2] = syllables[index2][1:]
                    syllables[index1][-1] = affr
                elif syl2[-2:] == [plosive_part]+[fricative_part]:
                    syllables[index2] = syllables[index2][:-2]
                    syllables[index2] = syllables[index2] + [affr]
                elif syl2[:2] == [plosive_part]+[fricative_part]:
                    syllables[index2] = [affr] + syl2[2:]
                # Combine split affricates in initial syllable
                elif syl1[:2] == [plosive_part]+[fricative_part]:
                    syllables[index1] = [affr] + syl1[2:]

    # Initialize syllables as Syllable class objects
    syllables = {i:Syllable(syllables[i], g_open=g_open) for i in syllables}

    return syllables


def sylTypes(word, **kwargs):
    """Returns a dictionary of syllables in a word and their type (open/closed)"""
    syllables = syllabify(word, **kwargs)
    return {i:(syllables[i], syllables[i].type) for i in syllables}


def scoreSyl(syl,
             max_onset=2,
             max_coda=2,
             illegal_coda=[],
             illegal_onset=[],
             no_onset_penalty=5,
             coda_penalty=2,
             complex_onset_penalty=1, # per extra segment in onset
             complex_coda_penalty=2,  # per extra segment in coda
             geminate_onset_penalty=3,
             exceeded_max_onset_penalty=5,
             exceeded_max_coda_penalty=5,
             sonority_violation_penalty=5,
             illegal_onset_penalty=5,  # per illegal onset segment
             illegal_coda_penalty=5,  # per illegal coda segment
             ):
    """Evaluates the degree of ill-formedness of a syllable according to various criteria.
    Penalizes:
    - no onset
    - complex onset
    - coda
    - exceeding max onset or coda length
    - illegal segments in onset or coda
    - sonority hierarchy violations
    """

    penalty = 0
    # syl, matched_affricates = split_affricates(''.join(syl))
    # syl = segment_ipa(syl)

    if type(syl) == list:
        onset, nucleus, coda = splitSyl(syl)
    elif type(syl) == str:
        onset, nucleus, coda = splitSyl(segment_ipa(syl))
    elif type(syl) == Syllable:
        onset, nucleus, coda = syl.onset, syl.nucleus, syl.coda
    else:
        raise ValueError

    # NO ONSET PENALTY (default = 5)
    if len(onset) < 1:
        penalty += no_onset_penalty
    else:
        onset_length = len(onset)
        # Affricates in onset count as if they were two consonants
        onset_length += sum([1 for o in onset if o in affricates])
        # COMPLEX ONSET PENALTY = default 1 * additional onset length
        if onset_length > 1:
            penalty += (onset_length * complex_onset_penalty) - 1

            # PENALIZE GEMINATES IN ONSET (default = 3)
            for i in range(1, len(onset)):
                if onset[i] == onset[i-1]:
                    penalty += geminate_onset_penalty

        # EXCEEDED MAX ONSET LENGTH PENALTY (default = 5)
        if onset_length > max_onset:
            penalty += exceeded_max_onset_penalty

        # ILLEGAL SEGMENTS/SEQUENCES IN ONSET (default = 5 each)
        for Xonset in illegal_onset:
            if re.search(Xonset, ''.join(onset)):
                penalty += illegal_onset_penalty

        # SONORITY HIERARCHY VIOLATION PENALTY (default = 5)
        # Sonority should increase from the left edge of the syllable toward the nucleus
        # Also penalize clusters of equal sonority in onset
        if onset_length > 1:
            onset_son = [_toSegment(o).get_sonority() for o in onset]
            for i in range(1, len(onset_son)):
                if onset_son[i] <= onset_son[i-1]:
                    penalty += sonority_violation_penalty

    # CODA PENALTY (default = 2)
    coda_length = len(coda)
    # Affricates in coda count as if they were two consonants
    # coda_length += sum([2 for c in coda if c in affricate])
    if coda_length > 0:
        penalty += coda_penalty

        # COMPLEX CODA PENALTY (default = 2 * additional length of coda)
        # (-1 because having a coda at all is already penalized)
        if coda_length > 1:
            penalty += complex_coda_penalty * (coda_length - 1)

        # EXCEEDED MAX CODA LENGTH PENALTY (default = 5)
        if coda_length > max_coda:
            penalty += exceeded_max_coda_penalty

        # ILLEGAL SEGMENTS/SEQUENCES IN CODA: 5 each
        for Xcoda in illegal_coda:
            if re.search(Xcoda, ''.join(coda)):
                penalty += illegal_coda_penalty

        # SONORITY HIERARCHY VIOLATION PENALTY (default = 5)
        # Sonority should decrease toward the right edge of a syllable
        coda_son = [_toSegment(c).get_sonority() for c in coda]
        if sorted(coda_son, reverse=True) != coda_son:
            penalty += sonority_violation_penalty

    return penalty


def openSyl(syl):
    return sylType(syl) == 'OPEN'


def closedSyl(syl):
    if sylType(syl) == 'CLOSED':
        return True
    else:
        return False