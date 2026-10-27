# Rule-guided poem generation.
#
# The model alone only imitates the surface of Devkota's style. Here each line is
# generated as N candidates, every candidate is scored against poem rules
# (rhythm, rhyme, repetition, simile, theme) plus the model's own fluency, and
# the best one is kept. No retraining needed; the rules live in PoemRules.
import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- devanagari

CONSONANTS = set(chr(c) for c in range(0x0915, 0x093A))     # क .. ह
VOWELS = set(chr(c) for c in range(0x0904, 0x0915))         # अ .. औ (independent vowels)
VIRAMA = "्"                                           # ्  (joins consonants)
NUKTA = "़"
PUNCT = re.compile(r"[\s!?,।॥.;:'\"‘’“”()\-–—]+$")
PUNCT_ANY = re.compile(r"[\s!?,।॥.;:'\"‘’“”()\-–—]+")

# stanza markers like "क." / "३." that older checkpoints learned from the data
MARKER = re.compile(r"(^|\s)([\u0915-\u0939]|[०-९0-9]+)\.")
MAX_PUNCT = 3                       # more "!"/"?" than this is the model padding the metre

SIMILE_WORDS = ("झैं", "झैँ", "झै", "जस्तै", "जस्तो", "जस्ती", "सरि", "सरी", "समान", "तुल्य")


def aksharas(text):
    """Split devanagari text into aksharas (written syllables).

    An akshara is one vowel nucleus: an independent vowel, or a consonant cluster
    (क्ष, स्त्र ...) with its vowel sign and any ं ँ ः. A consonant with a trailing
    virama at word end (जगत्) has no vowel, so it attaches to the previous akshara.
    """
    units, cur = [], ""
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in VOWELS:
            if cur:
                units.append(cur)
            cur = ch
        elif ch in CONSONANTS:
            # starts a new akshara unless the previous char was a virama (cluster)
            if cur and not cur.endswith(VIRAMA):
                units.append(cur)
                cur = ""
            cur += ch
        elif "़" <= ch <= "ॏ" or ch in "ँंः":
            cur += ch  # vowel signs, virama, nukta, ं ँ ः
        else:
            if cur:
                units.append(cur)
                cur = ""
        i += 1
    if cur:
        units.append(cur)
    # a bare halanta consonant (जगत् -> त्) is not a syllable; merge it back
    merged = []
    for u in units:
        if merged and u.endswith(VIRAMA) and not any(c in VOWELS for c in u):
            merged[-1] += u
        else:
            merged.append(u)
    return merged


def syllable_count(line):
    return len(aksharas(line))


# ---------------------------------------------------------------- metre (छन्द)

LONG_SIGNS = set("ाीूेैोौ")                 # vowel signs that make an akshara heavy
LONG_VOWELS = set("आईऊएऐओऔ")
HEAVY_MARKS = set("ंः")                     # anusvara, visarga (ँ chandrabindu stays light)

# L = लघु (light), G = गुरु (heavy); the last syllable of a line may be either
METRES = {
    "panchachamar": "LG" * 8,               # पञ्चचामर: ज र ज र ज ग, 16 syllables
}


def _starts_with_cluster(akshara):
    """True for a real consonant cluster (क्ष, न्द, ष्य ...), not a word-final halanta (द्).

    Nepali poets sometimes read a cluster as not lengthening the previous akshara
    (बन्यौ, रङ्गीन); that freedom is why the generator allows a small metre error.
    """
    a = akshara.replace(NUKTA, "")
    return len(a) > 2 and a[0] in CONSONANTS and a[1] == VIRAMA and a[2] in CONSONANTS


def weights(line):
    """लघु/गुरु pattern of a line, e.g. 'LGLG...'.

    An akshara is heavy (G) if it has a long vowel, ं or ः, ends in a halanta (शरद्),
    or is followed inside the same word by a consonant cluster (प्रा|प्ति -> प्रा is G).
    """
    out = []
    for word in PUNCT_ANY.split(line):
        units = aksharas(word)
        for k, u in enumerate(units):
            heavy = (
                any(c in LONG_SIGNS or c in HEAVY_MARKS for c in u)
                or u[0] in LONG_VOWELS
                or u.endswith(VIRAMA)
                or (k + 1 < len(units) and _starts_with_cluster(units[k + 1]))
            )
            out.append("G" if heavy else "L")
    return "".join(out)


def metre_errors(line, pattern):
    """Positions that break the metre, plus any missing or extra syllables."""
    w = weights(line)
    errors = abs(len(w) - len(pattern))
    # final syllable is free (anceps), as in sanskrit/nepali prosody
    errors += sum(a != b for a, b in zip(w[:len(pattern) - 1], pattern[:len(pattern) - 1]))
    return errors


def rhyme_key(line, n=1):
    """Last n aksharas of the line, ignoring punctuation: the sound that must rhyme."""
    line = PUNCT.sub("", line.strip())
    words = line.split()
    if not words:
        return ""
    return "".join(aksharas(words[-1])[-n:])


def scheme_partner(scheme, i):
    """Index of the earlier line that line i must rhyme with, or None.

    scheme repeats per stanza, e.g. "AABB" or "ABAB"; "" means no rhyme.
    """
    if not scheme:
        return None
    k = len(scheme)
    stanza_start, pos = (i // k) * k, i % k
    for j in range(pos - 1, -1, -1):
        if scheme[j] == scheme[pos]:
            return stanza_start + j
    return None


# ---------------------------------------------------------------- rules

@dataclass
class PoemRules:
    lines: int = 8
    metre: str | None = None         # name in METRES or a raw L/G pattern; enforced while sampling
    metre_tolerance: int = 1         # metre errors allowed per line (nepali prosody has some freedom)
    syllables: int | None = None     # target aksharas per line, used when no metre is set
    syllable_tolerance: int = 1      # +/- allowed before penalties kick in
    rhyme: str = ""                  # per-stanza scheme like "AABB" or "ABAB"; "" disables rhyme
    rhyme_aksharas: int = 1          # how many final aksharas must match
    min_words: int = 3               # reject one-word "?" lines
    theme_words: list = field(default_factory=list)

    # weights of each rule in the score (fluency is the model's mean log-prob, ~ -2..-6)
    w_metre: float = 2.0             # per metre error that survived sampling
    w_rhythm: float = 1.0            # per syllable outside tolerance
    w_rhyme: float = 3.0             # bonus when a required rhyme matches
    w_repeat: float = 5.0            # penalty for repeating words of earlier lines
    w_simile: float = 1.0            # bonus for झैं / जस्तै ... (metaphor proxy)
    w_theme: float = 1.0             # bonus per theme word used

    @property
    def pattern(self):
        if not self.metre:
            return None
        return METRES.get(self.metre, self.metre.upper())


def cheats(line):
    """Hard-rule breaks: markers, punctuation padding, a word used 3+ times."""
    if MARKER.search(line) or sum(line.count(c) for c in "!?") > MAX_PUNCT:
        return True
    words = PUNCT_ANY.split(line)
    return any(words.count(w) > 2 for w in set(words) if w)


def score_line(line, prev_lines, i, rules, fluency):
    words = PUNCT.sub("", line).split()
    if len(words) < rules.min_words or cheats(line):
        return float("-inf")
    if line.strip() in (p.strip() for p in prev_lines):
        return float("-inf")

    score = fluency

    if rules.pattern:
        score -= rules.w_metre * metre_errors(line, rules.pattern)
    elif rules.syllables:
        off = abs(syllable_count(line) - rules.syllables)
        score -= rules.w_rhythm * max(0, off - rules.syllable_tolerance)

    partner = scheme_partner(rules.rhyme, i)
    if partner is not None and partner < len(prev_lines):
        want = rhyme_key(prev_lines[partner], rules.rhyme_aksharas)
        if want and rhyme_key(line, rules.rhyme_aksharas) == want:
            score += rules.w_rhyme
        # the same last word is not a rhyme, it is repetition
        if words and PUNCT.sub("", prev_lines[partner]).split()[-1:] == words[-1:]:
            score -= rules.w_rhyme

    # share of this line's words already used in the last few lines
    recent = set(w for p in prev_lines[-4:] for w in PUNCT.sub("", p).split())
    if recent:
        score -= rules.w_repeat * sum(w in recent for w in words) / len(words)
    # and words repeated inside the line itself
    plain = PUNCT_ANY.split(line)
    score -= rules.w_repeat * (len(plain) - len(set(plain))) / max(len(plain), 1)

    if any(s in line for s in SIMILE_WORDS):
        score += rules.w_simile
    score += rules.w_theme * sum(t in line for t in rules.theme_words)
    return score


def metre_checker(pattern, tolerance, decode, newline_id, eos_id):
    """Returns check(line_ids, token) -> bool, used to reject tokens while sampling.

    A token is allowed if the line so far still fits the L/G pattern (the last
    akshara is skipped: a following cluster or vowel sign can still change it),
    and the line may only end once it has exactly len(pattern) aksharas.
    """
    n = len(pattern)

    def check(line_ids, tok):
        if tok == eos_id:
            return False
        if tok == newline_id:
            line = decode(line_ids)
            return len(weights(line)) == n and metre_errors(line, pattern) <= tolerance
        text = decode(line_ids + [tok])
        if cheats(text):
            return False
        w = weights(text)
        if len(w) > n:
            return False
        return sum(a != b for a, b in zip(w[:-1], pattern)) <= tolerance

    return check


# ---------------------------------------------------------------- generation

@torch.no_grad()
def sample_lines(model, context, n, newline_id, eos_id, check=None, max_tokens=64,
                 temperature=0.8, top_k=50):
    """Beam search for up to n lines after context, each ending at a newline.

    Rejection sampling picks random tokens that fit the metre, which with a small
    model gives nonsense; a beam keeps the n most likely prefixes that still fit.
    check(line_ids, token) -> bool rejects tokens (e.g. ones that break the metre).
    Gumbel noise scaled by temperature varies the beam so poems differ per run.
    Returns (token_ids, mean_logprob) for every line that finished.
    """
    device = next(model.parameters()).device
    beams, finished = [([], 0.0)], []
    for _ in range(max_tokens):
        idx = torch.tensor([context + ids for ids, _ in beams], dtype=torch.long, device=device)
        logits = model(idx[:, -model.max_seq_len:], return_loss=False)["logits"][:, -1]
        top_lp, top_tok = F.log_softmax(logits, dim=-1).topk(top_k, dim=-1)

        grown = []
        for (ids, lp), lps, toks in zip(beams, top_lp.tolist(), top_tok.tolist()):
            for l, t in zip(lps, toks):
                if t == eos_id or (check is not None and not check(ids, t)):
                    continue
                if t == newline_id:
                    if ids:
                        finished.append((ids, (lp + l) / (len(ids) + 1)))
                    continue
                noise = -temperature * torch.empty(1).exponential_().log().item()
                grown.append((lp + l + noise, ids + [t], lp + l))
        if not grown or len(finished) >= n:
            break
        grown.sort(key=lambda g: g[0], reverse=True)
        beams = [(ids, lp) for _, ids, lp in grown[:n]]
    return finished


def write_poem(model, sp, rules: PoemRules, prompt="", candidates=24,
               temperature=0.8, top_k=50, stanza=None, verbose=False):
    """Generate a poem line by line, keeping the best-scoring candidate each time."""
    model.eval()
    newline_id, eos_id = sp.piece_to_id("<0x0A>"), sp.eos_id()
    check = None
    if rules.pattern:
        check = metre_checker(rules.pattern, rules.metre_tolerance, sp.decode, newline_id, eos_id)

    # the prompt (a title or opening line) stays in the context for every line
    context = sp.encode(prompt) + [newline_id] if prompt else [newline_id]
    lines = []
    for i in range(rules.lines):
        best, best_score, best_ids = None, float("-inf"), None
        for attempt in range(3):  # retry if no candidate survived the rules
            for ids, fluency in sample_lines(model, context, candidates, newline_id, eos_id,
                                             check=check, temperature=temperature, top_k=top_k):
                text = sp.decode(ids).strip()
                s = score_line(text, lines, i, rules, fluency)
                if s > best_score:
                    best, best_score, best_ids = text, s, ids
            if best is not None:
                break
        if best is None:
            raise RuntimeError(f"line {i + 1}: no candidate fit the rules; "
                               f"raise --candidates or --metre-tolerance")
        if verbose:
            extra = f"metre errors {metre_errors(best, rules.pattern)}" if rules.pattern \
                else f"{syllable_count(best)} aksharas"
            print(f"  [{i + 1}] score {best_score:.2f} | {extra} | {weights(best)} | "
                  f"rhyme '{rhyme_key(best, rules.rhyme_aksharas)}'")
        lines.append(best)
        context = context + best_ids + [newline_id]

    stanza = stanza or (len(rules.rhyme) if rules.rhyme else 4)
    out = [prompt] if prompt else []
    for i, line in enumerate(lines):
        if i and i % stanza == 0:
            out.append("")
        out.append(line)
    return "\n".join(out)
