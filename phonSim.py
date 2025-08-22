# PHONETIC SEGMENT ANALYSIS AND PHONETIC SIMILARITY/DISTANCE
# Code developed by Philip Georgis (Last updated: August 2023)

import os
import sys
from collections import defaultdict
from functools import lru_cache

from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phonUtils.constants import FEATURE_WEIGHTS, IPA_SEGMENTS
from phonUtils.segment import _toSegment


# SIMILARITY / DISTANCE MEASURES
def hamming_distance(vec1, vec2, normalize=True):
    differences = len([feature for feature in vec1 if vec1[feature] != vec2[feature]])
    if normalize:
        return differences / len(vec1)
    else: 
        return differences


def jaccard_sim(vec1, vec2):
    features = sorted(list(vec1.keys()))
    vec1_values = [vec1[feature] for feature in features]
    vec2_values = [vec2[feature] for feature in features]
    
    # Jaccard index does not allow continuous features
    # Ensure that they are all binarily encoded (any continuous value >0 --> 1)
    for vec in [vec1_values, vec2_values]:
        for i in range(len(vec)):
            if vec[i] > 0:
                vec[i] = 1
                
    return jaccard_score(vec1_values, vec2_values)


def dice_sim(vec1, vec2):
    jaccard = jaccard_sim(vec1, vec2)
    return (2*jaccard) / (1+jaccard)


def weighted_hamming(vec1, vec2, weights=FEATURE_WEIGHTS):
    diffs = 0
    for feature in vec1:
        if vec1[feature] != vec2[feature]:
            diffs += weights[feature]
    return diffs/len(vec1)


def weighted_jaccard(vec1, vec2, weights=FEATURE_WEIGHTS):
    union, intersection = 0, 0
    for feature in vec1:
        if ((vec1[feature] == 1) and (vec2[feature] == 1)):
            intersection += weights[feature]
        if ((vec1[feature] == 1) or (vec2[feature] == 1)):
            union += weights[feature]
    return intersection/union
            

def weighted_dice(vec1, vec2, weights=FEATURE_WEIGHTS):
    w_jaccard = weighted_jaccard(vec1, vec2, weights)
    return (2*w_jaccard) / (1+w_jaccard)


# PHONE COMPARISON
@lru_cache(maxsize=None)
def phone_sim(phone1, phone2, similarity='weighted_dice', exclude_features=None):
    """Returns the similarity of the features of the two phones according to
    the specified distance/similarity function;
    Features not to be included in the comparison should be passed as a list to
    the exclude_features parameter (by default no features excluded)"""

    if exclude_features is None:
        exclude_features = set()

    # Convert IPA strings to Segment objects
    phone1, phone2 = map(_toSegment, [phone1, phone2])

    # Get feature dictionaries for each phone
    phone_id1, phone_id2 = phone1.features.copy(), phone2.features.copy()

    # Exclude specified features
    for feature in exclude_features:
        phone_id1.pop(feature, None)
        phone_id2.pop(feature, None)

    # Calculate similarity of phone features according to specified measure
    if similarity in ['cosine', 'weighted_cosine']:
        compare_features = list(phone_id1.keys())
        phone1_values = [phone_id1[feature] for feature in compare_features]
        phone2_values = [phone_id2[feature] for feature in compare_features]
        if similarity == 'weighted_cosine':
            weights = [FEATURE_WEIGHTS[feature] for feature in compare_features]
            # Subtract from 1: cosine() returns a distance
            score = 1 - cosine(phone1_values, phone2_values, w=weights)
        else:
            score = 1 - cosine(phone1_values, phone2_values)
    else:
        measure = {
            'dice': dice_sim,
            'hamming': hamming_distance,
            'jaccard': jaccard_sim,
            'weighted_dice': weighted_dice,
            'weighted_hamming': weighted_hamming,
            'weighted_jaccard': weighted_jaccard
        }.get(similarity)
        if measure is None:
            raise KeyError(f'Error: similarity measure "{similarity}" not recognized!')
        score = measure(phone_id1, phone_id2)

    # If method is Hamming, convert distance to similarity
    if similarity in ['hamming', 'weighted_hamming']:
        score = 1 - score

    return score


# Helper functions for identifying phones with particular features
def lookup_segments(features, values, segment_list=IPA_SEGMENTS):
    """Returns a list of segments whose feature values match the search criteria"""
    matches = []
    for segment in segment_list:
        segment = _toSegment(segment)
        match_tallies = 0
        for feature, value in zip(features, values):
            if segment.features[feature] == value:
                match_tallies += 1
        if match_tallies == len(features):
            matches.append(segment.segment)
    return set(matches)


def common_features(segment_list, start_features=FEATURE_WEIGHTS.keys()):
    """Returns the features/values shared by all segments in the list"""
    features = set(start_features)
    feature_values = defaultdict(lambda:[])
    for segment in segment_list:
        segment = _toSegment(segment)
        for feature in features:
            value = segment.features[feature]
            if value not in feature_values[feature]:
                feature_values[feature].append(value)
    common = [(feature, feature_values[feature][0]) for feature in feature_values if len(feature_values[feature]) == 1]
    return common
