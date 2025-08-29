from phonTransforms import (degeminate, finalDevoicing, normalize_geminates,
                            regressiveVoicingAssimilation, shiftAccent,
                            shiftStress, split_affricates,
                            unstressedVowelReduction)


def test_degeminate():
    test_ref_pairs = {
        "sˈappja": "sˈapja",
        "sapːja": "sapja",
        "a kkˈaza": "a kˈaza",
        "non so kkˈe kːˈɔza": "non so kˈe kˈɔza",
        "mukkʰ": "mukʰ",
        "mukʰː": "mukʰ",
        "mississippi": "misisipi",
    }
    
    # Default degemination of all consonants 
    for str, ref in test_ref_pairs.items():
        assert degeminate(str) == ref
    
    # Degemination of only specific consonants 
    assert degeminate("mississippi", phones={"s"}) == "misisippi"


def test_normalize_geminates():
    test_ref_pairs = {
        "sˈappja": "sˈapːja",
        "sapːja": "sapːja",
        "a kkˈaza": "a kːˈaza",
        "non so kkˈe kːˈɔza": "non so kːˈe kːˈɔza",
        "mukkʰ": "mukʰː",
        "mukʰː": "mukʰː",
        "mississippi": "misːisːipːi",
        "ɑːʰttɑ": "ɑːʰtːɑ",
    }
    for str, ref in test_ref_pairs.items():
        assert normalize_geminates(str) == ref


def test_finalDevoicing():
    test_ref_pairs = {
        "hund": "hunt",
        "ʃtaːb": "ʃtaːp",
        "staʐ": "staʂ",
        "θað": "θaθ",
        "vjaʤ": "vjaʧ",
        "berɡ": "berk",
    }

    # Default settings
    for str, ref in test_ref_pairs.items():
        assert finalDevoicing(str) == ref

    # Using custom devoice dict
    devoice_dict = {'b': 'b̥', 'd':'d̥', 'ɡ':'ɡ̊', 'ʐ':'ʐ̊'}
    test_ref_pairs = {
        "hund": "hund̥",
        "ʃtaːb": "ʃtaːb̥",
        "staʐ": "staʐ̊",
        "berɡ": "berɡ̊",
    }
    for str, ref in test_ref_pairs.items():
        assert finalDevoicing(str, devoice_dict=devoice_dict) == ref
    