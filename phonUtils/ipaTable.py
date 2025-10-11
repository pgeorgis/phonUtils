import re
from texttable import Texttable
from typing import Iterable

from .segment import Segment

MANNERS = [
    "PLOSIVE",
    "IMPLOSIVE",
    "AFFRICATE",
    "FRICATIVE",
    "LATERAL FRICATIVE",
    "NASAL",
    "FRICATIVE TRILL",
    "TRILL",
    "TAP/FLAP",
    "LATERAL APPROXIMANT",
    "APPROXIMANT",
    "CLICK",
    "DIPHTHONG",
    "VOWEL",
    "TONEME",
    "SUPRASEGMENTAL",
]

PLACES_OF_ARTICULATION = [
    "BILABIAL",
    "LABIAL-VELAR",
    "LABIODENTAL",
    "DENTAL",
    "LINGUO-LABIAL",
    "APICO-ALVEOLAR",
    "LAMINAL ALVEOLAR",
    "ALVEOLAR",
    "LATERAL",  # should be manner: lateral approximant
    "POSTALVEOLAR",
    "ALVEOLOPALATAL",
    "RETROFLEX",
    "PALATAL",
    "VELAR",
    "UVULAR",
    "PHARYNGEAL",
    "EPIGLOTTAL",
    "GLOTTAL",
]

def build_consonant_table(consonant_set: Iterable[Segment | str], mode='ascii'):
    consonant_set: set[Segment] = set(Segment(seg) if isinstance(seg, str) else seg for seg in consonant_set)
    for seg in consonant_set:
        if seg.phone_class not in ("CONSONANT", "GLIDE"):
            raise ValueError(f"Segment <{seg.segment}> is not a consonant")

    # Get attested places of articulation and manners from among the input character set
    attested_poa = set(s.poa for s in consonant_set)
    attested_manner = set(s.manner for s in consonant_set)
    sorted_attested_poa = [p for p in PLACES_OF_ARTICULATION if p in attested_poa]
    sorted_attested_manner = [m for m in MANNERS if m in attested_manner]

    # Fill table with place and manner of articulation of given segments
    table = {m: {p: [] for p in sorted_attested_poa} for m in sorted_attested_manner}
    for segment in consonant_set:
        manner, place = segment.manner, segment.poa
        table[manner][place].append(segment)
        
    
    # Sort entries within same table cell
    # Voiceless on left, voiced on right
    def _is_devoiced(seg):
        return re.search(r"[̥̊]", seg.segment) is not None

    def _voicing_rank(seg):
        # 0 = voiceless, 1 = devoiced, 2 = voiced
        if seg.voiced:
            return 2
        elif _is_devoiced(seg):
            return 1
        else:
            return 0

    for manner in table:
        for place in table[manner]:
            if len(table[manner][place]) > 1:
                table[manner][place].sort(key = lambda seg: (_voicing_rank(seg), seg.base, seg.segment))
            table[manner][place] = ", ".join([seg.segment for seg in table[manner][place]])


    try:
        t = Texttable(max_width=0)
        t.header([""] + sorted_attested_poa)
        for manner in sorted_attested_manner:
            t.add_row([manner] + [table[manner][p] for p in sorted_attested_poa])
        return t.draw()
    except ImportError:
        # fallback: simple text table
        header = ["{:>15}".format(p) for p in [""] + sorted_attested_poa]
        lines = [" ".join(header)]
        for manner in sorted_attested_manner:
            row = ["{:>15}".format(manner)] + ["{:>15}".format(table[manner][p]) for p in sorted_attested_poa]
            lines.append(" ".join(row))
        return "\n".join(lines)

