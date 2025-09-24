from phonUtils.phonTransforms import (VOICED_CONSONANTS, degeminate,
                                      finalDevoicing, normalize_geminates,
                                      regressiveVoicingAssimilation,
                                      shiftAccent, shiftStress,
                                      split_affricates,
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


def test_split_affricates():
    test_ref_pairs = {
        'kaʦə': 'katsə',
        'ʣel': 'dzel',
        'ʧokolada': 'tʃokolada',
        'aʤʤo': 'adʒdʒo',
        'bɨʨ': 'bɨtɕ',
        'muʥna': 'mudʑna'
    }

    for str, ref in test_ref_pairs.items():
        split_affr, _ = split_affricates(str)
        assert split_affr == ref


def test_shiftStress():
    test_ref_pairs = {
        ("ʧokolˈada", 0): "ʧˈokolada",
        ("ʧokolˈada", 1): "ʧokˈolada",
        ("ʧokolˈada", -1): "ʧokoladˈa",
        ("ʧokolˈada", -2): "ʧokolˈada",
    }

    for (str, syln), ref in test_ref_pairs.items():
        assert shiftStress(str, n_syl=syln) == ref

    # Test secondary stress shift
    assert shiftStress("ʧokolˈada", n_syl=0, type='SECONDARY') == "ʧˌokolˈada"


def test_shiftAccent():
    test_ref_pairs = {
        ("rjeːkǎ", "̌", 0): "rjěːka",
        ("móri", "́", -1): "morí",
        ("lâbeno", "̂", -2): "labêno",
    }

    for (str, ch, syln), ref in test_ref_pairs.items():
        assert shiftAccent(str, n_syl=syln, accent_ch=ch) == ref


def test_unstressedVowelReduction():
    test_ref_pairs = {
        "plˈante": "plˈantə",
        "katalˈa": "kətəlˈa",
        "molokˈo": "mələkˈo",
        "mˈøy̯zi": "mˈøə̯zə",
    }
    for string, ref in test_ref_pairs.items():
        assert unstressedVowelReduction(string) == ref

    # Disable diphthong reduction
    assert unstressedVowelReduction("mˈøy̯zi", reduce_diphthongs=False) == "mˈøy̯zə"

    # Reduce only specific vowels
    test_ref_pairs = {
        "plˈante": "plˈante",
        "katalˈa": "kətəlˈa",
        "molokˈo": "molokˈo",
        "mˈøy̯zi": "mˈøy̯zi",
    }
    for string, ref in test_ref_pairs.items():
        assert unstressedVowelReduction(string, vowels={'a'}) == ref

    # Specify different reduction target
    test_ref_pairs = {
        "plˈante": "plˈante",
        "katalˈa": "kɐtɐlˈa",
        "molokˈo": "molokˈo",
        "mˈøy̯zi": "mˈøy̯zi",
    }
    for string, ref in test_ref_pairs.items():
        assert unstressedVowelReduction(string, vowels={'a'}, reduced='ɐ') == ref

    # Custom vowel reduction mapping
    reduction_dict = {
        'a': 'ɐ',
        'o': 'u',
        'e': 'ə',
        'i': 'e',
    }
    test_ref_pairs = {
        "plˈante": "plˈantə",
        "katalˈa": "kɐtɐlˈa",
        "molokˈo": "mulukˈo",
        "mˈøy̯zi": "mˈøy̯ze",
    }
    for string, ref in test_ref_pairs.items():
        assert unstressedVowelReduction(string, reduction_dict=reduction_dict) == ref


def test_regressiveVoicingAssimilation():
    test_ref_pairs = {
        "kdo": "ɡdo",
        "hau̯bt": "hau̯pt",
        "vezds": "vests",
        "laʃpmi": "laʒbmi",
        "ʤupro": "ʤubro",
        "saxla": "saɣla",
        "sofna": "sovna",
    }
    for string, ref in test_ref_pairs.items():
        assert regressiveVoicingAssimilation(string) == ref

    # only voiced -> voiceless
    test_ref_pairs = {
        "kdo": "kdo",
        "hau̯bt": "hau̯pt",
        "vezds": "vests",
        "laʃpmi": "laʃpmi",
        "ʤupro": "ʤupro",
        "saxla": "saxla",
        "sofna": "sofna"
    }
    for string, ref in test_ref_pairs.items():
        assert regressiveVoicingAssimilation(string, to_voiced=False) == ref

    # only voiceless -> voiced
    test_ref_pairs = {
        "kdo": "ɡdo",
        "hau̯bt": "hau̯bt",
        "vezds": "vezds",
        "laʃpmi": "laʒbmi",
        "ʤupro": "ʤubro",
        "saxla": "saɣla",
        "sofna": "sovna",
    }
    for string, ref in test_ref_pairs.items():
        assert regressiveVoicingAssimilation(string, to_voiceless=False) == ref

    # test with exception
    # e.g. no assimilation triggered by /r, l, m, n/
    test_ref_pairs = {
        "kdo": "ɡdo",
        "hau̯bt": "hau̯pt",
        "vezds": "vests",
        "laʃpmi": "laʃpmi",
        "ʤupro": "ʤupro",
        "saxla": "saxla",
        "sofna": "sofna",
    }
    for string, ref in test_ref_pairs.items():
        assert regressiveVoicingAssimilation(string, exception=[rf'[{VOICED_CONSONANTS}][rlmn]']) == ref
