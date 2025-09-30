from functools import lru_cache

from scipy.spatial.distance import cosine
from sklearn.metrics import jaccard_score

from .constants import FEATURE_WEIGHTS
from .segment import Segment

MEASURES = {
    'cosine',
    'weightedCosine',
    'hamming',
    'weightedHamming',
    'dice',
    'weightedDice',
    'jaccard',
    'weightedJaccard',
}

def hamming_distance(vec1: dict,
                     vec2: dict,
                     normalize: bool = True
                     ) -> float | int:
    differences = len([feature for feature in vec1 if vec1[feature] != vec2[feature]])
    if normalize:
        return differences / len(vec1)
    else: 
        return differences


def jaccard_sim(vec1: dict,
                vec2: dict
                ) -> float:
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


def dice_sim(vec1: dict,
             vec2: dict
             ) -> float:
    jaccard = jaccard_sim(vec1, vec2)
    return (2*jaccard) / (1+jaccard)


def weighted_hamming(vec1: dict,
                     vec2: dict,
                     weights: dict = FEATURE_WEIGHTS
                     ) -> float:
    diffs = 0
    for feature in vec1:
        if vec1[feature] != vec2[feature]:
            diffs += weights[feature]
    return diffs/len(vec1)


def weighted_jaccard(vec1: dict,
                     vec2: dict,
                     weights: dict = FEATURE_WEIGHTS
                     ) -> float:
    union, intersection = 0, 0
    for feature in vec1:
        if ((vec1[feature] == 1) and (vec2[feature] == 1)):
            intersection += weights.get(feature, 0)
        if ((vec1[feature] == 1) or (vec2[feature] == 1)):
            union += weights.get(feature, 0)
    if union == 0:
        return 0.0
    return intersection / union
            

def weighted_dice(vec1: dict,
                  vec2: dict,
                  weights: dict = FEATURE_WEIGHTS
                  ) -> float:
    w_jaccard = weighted_jaccard(vec1, vec2, weights)
    if w_jaccard == 0:
        return 0.0
    return (2 * w_jaccard) / (1 + w_jaccard)


@lru_cache(maxsize=None)
def phone_sim(phone1: str,
              phone2: str,
              measure: str = 'weighted dice',
              exclude_features: set = None
              ):
    """Returns the similarity of the features of the two phones according to
    the specified distance/similarity function;
    Features not to be included in the comparison should be passed as a list to
    the exclude_features parameter (by default no features excluded)"""
    if measure not in MEASURES:
        raise ValueError(f"Unrecognized similarity/distance measure <{measure}>")

    if exclude_features is None:
        exclude_features = set()

    # Convert IPA strings to Segment objects
    phone1, phone2 = map(Segment, [phone1, phone2])

    # Get feature dictionaries for each phone
    phone_id1, phone_id2 = phone1.features.copy(), phone2.features.copy()

    # Exclude specified features
    for feature in exclude_features:
        phone_id1.pop(feature, None)
        phone_id2.pop(feature, None)

    # Calculate similarity of phone features according to specified measure
    if measure in ['cosine', 'weightedCosine']:
        compare_features = list(phone_id1.keys())
        phone1_values = [phone_id1[feature] for feature in compare_features]
        phone2_values = [phone_id2[feature] for feature in compare_features]
        if measure == 'weightedCosine':
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
            'weightedDice': weighted_dice,
            'weightedHamming': weighted_hamming,
            'weightedJaccard': weighted_jaccard
        }.get(measure)
        score = measure(phone_id1, phone_id2)

    # If method is Hamming, convert distance to similarity
    if measure in ['hamming', 'weightedHamming']:
        score = 1 - score

    return score
