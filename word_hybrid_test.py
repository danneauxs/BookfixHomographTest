"""
Homograph Disambiguation Test — TTS pronunciation resolver.

Purpose: Determines correct pronunciation spelling for homographs (words spelled the same, sound different)
         to prepare text for Text-to-Speech engines.

Method: Hybrid approach uses SpaCy POS/dependency parsing first, RoBERTa-large-mnli NLI for ambiguous cases.

Inputs:
    - InputText/*.txt           # Text files to analyze (selected at runtime)
    - data/choices.json        # Homograph definitions (verb vs noun spellings)
    - hybrid_replacements.txt   # Preprocessing rules (find/replace before analysis)

Outputs:
    - *_hybrid_results.txt   # Prediction results with POS, dependency, route info
    - OutputText/*_processed.txt  # Production mode: replaced homograph text

Required models:
    - spacy: en_core_web_trf    (Transformer-based POS/dependency)
    - transformers: roberta-large-mnli  (NLI for disambiguation)

Usage:
    source venv/bin/activate
    python test/word_hybrid_test.py record
    python test/word_hybrid_test.py close
    python test/word_hybrid_test.py
"""

import re
import sys
import json
from pathlib import Path
from transformers import pipeline as hf_pipeline
import spacy
import torch

# ── Configuration ────────────────────────────────────────────────────────────────────────
# Models: NLI for disambiguation, POS/dependency via SpaCy
NLI_MODEL    = "roberta-large-mnli"
POS_MODEL    = "en_core_web_trf"
DATA_DIR     = Path(__file__).parent / "1" / "data"
OUT_DIR     = Path(__file__).parent
REPLACE_FILE = Path(__file__).parent / "hybrid_replacements.txt"

# ── POS Tag Categories ────────────────────────────────────────────────────────────────────────
# Used to classify word's grammatical role for disambiguation.
VERB_PRESENT = {"VB", "VBP", "VBZ", "VBG", "VBN"}
VERB_PAST    = {"VBD"}
ADJ_TAGS    = {"JJ", "JJR", "JJS"}
ADV_TAGS    = {"RB", "RBR", "RBS"}


# ── Data Loading ────────────────────────────────────────────────────────────────────────────────

def load_choices():
    """Load data/choices.json — returns dict {word: [options]}."""
    path = Path(__file__).parent.parent / "data" / "choices.json"
    return {e["word"].lower(): e["options"] for e in json.load(open(path))}


def load_replacements():
    """Load hybrid_replacements.txt — returns list of (pattern, replacement) tuples."""
    """Load data/replace.txt into a list of (pattern, replacement) tuples."""
    rules = []
    if not REPLACE_FILE.exists():
        return rules
    for raw in REPLACE_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or " -> " not in line:
            continue
        lhs, rhs = line.split(" -> ", 1)
        if lhs.startswith("literal:"):
            pat = re.compile(re.escape(lhs[8:]), re.IGNORECASE)
        elif lhs.startswith("regex:"):
            pat = re.compile(lhs[6:], re.IGNORECASE)
        else:
            pat = re.compile(re.escape(lhs), re.IGNORECASE)
        rules.append((pat, rhs))
    return rules


def apply_replacements(text, rules):
    """Apply find/replace rules to text — returns modified text."""
    for pat, rhs in rules:
        text = pat.sub(rhs, text)
    return text


# ── Spelling Helpers ──────────────────────────────────────────────────────────────────

def get_spellings(word, options):
    """Extract verb_spelling and default_spelling from choices.json options."""
    """Return (verb_spelling, default_spelling) for the word."""
    verb_sp = None
    default_sp = None
    for opt in options:
        pos_parts = [p.strip().upper() for p in opt.get("pos", "").split("/")]
        defn = opt.get("definition", "").lower()
        is_verb = any(p == "VERB" for p in pos_parts)
        if not is_verb and not opt.get("pos"):
            # Infer from definition
            is_verb = any(k in defn for k in ["to coil", "to twist", "to wrap", "to shut",
                                               "to plant", "to decline", "to capture"])
        if is_verb:
            if verb_sp is None:
                verb_sp = opt["spelling"]
        else:
            if default_sp is None:
                default_sp = opt["spelling"]

    if verb_sp is None and default_sp is None:
        verb_sp = options[0]["spelling"]
        default_sp = options[1]["spelling"]
    elif verb_sp is None:
        verb_sp = next(o["spelling"] for o in options if o["spelling"] != default_sp)
    elif default_sp is None:
        default_sp = next(o["spelling"] for o in options if o["spelling"] != verb_sp)

    return verb_sp, default_sp


def split_sentences(text):
    # Split paragraphs into sentences on period, exclamation, question mark boundaries
    # Pattern handles optional quote marks after sentence-ending punctuation
    # Characters: ASCII quotes (34, 39) and smart quotes (8220, 8221, 8216, 8217)
    quotes = chr(34) + chr(39) + chr(8220) + chr(8221) + chr(8216) + chr(8217)
    pattern = "(?<=[.!?])[" + quotes + "]?\\s+(?=[A-Z" + quotes + "])"
    return [s.strip() for s in re.split(pattern, text) if s.strip()]


def get_token_info(nlp, sentence, word):
    doc = nlp(sentence)
    wl = word.lower()
    results = []
    for token in doc:
        if token.text.lower() == wl:
            has_det = any(c.dep_ == "det" for c in token.children)
            results.append((token.tag_, token.dep_, token.head.text.lower(), has_det))
    if not results:
        results.append(("UNKNOWN", "UNKNOWN", "UNKNOWN", False))
    return results


def nli_decide(nli, sentence, word, q_a, label_a, q_b, label_b):
    result = nli(sentence, candidate_labels=[q_a, q_b], multi_label=False)
    scores = dict(zip(result["labels"], result["scores"]))
    if scores[q_a] > scores[q_b]:
        return label_a, f"nli={label_a}({scores[q_a]:.2f})"
    else:
        return label_b, f"nli={label_b}({scores[q_b]:.2f})"


# ── POS-only decisions ────────────────────────────────────────────────────────

def decide_record(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "rekord", "pos-verb"
    return "rekkurd", "pos-noun"


def decide_close(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "cloze", "pos-verb"
    if tag in ADJ_TAGS or tag in ADV_TAGS:
        return "close", "pos-adj/adv"
    return "close", "pos-default"


def decide_read(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PAST or tag == "VBN":
        return "red", "pos-past"
    if tag in VERB_PRESENT:
        return "reed", "pos-verb"
    return "red", "pos-default"


def decide_live(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "liv", "pos-verb"
    if tag in ADJ_TAGS or tag in ADV_TAGS:
        return "lyve", "pos-adj/adv"
    return "liv", "pos-default"


def decide_object(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "ubjekt", "pos-verb"
    return "objekt", "pos-noun"


def decide_present(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "prezent", "pos-verb"
    return "present", "pos-noun/adj"


def decide_sow(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "soh", "pos-verb"
    return "sow", "pos-noun"


def decide_resume(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "rezoom", "pos-verb"
    return "resume", "pos-noun"


def decide_refuse(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "refuze", "pos-verb"
    return "refuse", "pos-noun"


def decide_elaborate(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "elaboreight", "pos-verb"
    return "elaborit", "pos-adj"


def decide_estimate(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "estimeight", "pos-verb"
    return "estimit", "pos-noun"


def decide_wind(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "why'nd", "pos-verb"
    return "win'd", "pos-noun"


def decide_invalid(tag, dep, head, has_det, nli, sentence):
    if tag in ADJ_TAGS:
        # "some kind of invalid" — article is on "kind" not "invalid", so has_det=False
        # but dep=pobj head=of still means noun-person sense
        if dep == "pobj" and head == "of":
            return "invalid", "jj-pobj-of-noun"
        # "the invalid" used as a noun-person — JJ with determiner in nsubj/dobj position
        if has_det and dep in {"nsubj", "dobj", "pobj", "nsubjpass"}:
            return "invalid", "jj-but-noun"
        return "in-valid", "pos-adj"
    return "invalid", "pos-noun"


_TEAR_EYES_AWAY = re.compile(
    r'\btear\b.{0,40}?\b(eyes?|gaze|attention|focus|sight)\b.{0,20}?\baway\b',
    re.IGNORECASE | re.DOTALL
)


def decide_tear(tag, dep, head, has_det, nli, sentence):
    if tag in VERB_PRESENT or tag in VERB_PAST:
        # "tear eyes/gaze/attention away" = physical pulling, not crying
        if _TEAR_EYES_AWAY.search(sentence):
            return "tair", "pos-tear-eyes-away"
        # Both senses have verb forms: "to tear (rip)" and "to tear (fill with tears)"
        # Use NLI to distinguish rather than assuming all verbs = rip
        return nli_decide(nli, sentence, "tear",
            "In this sentence, the word 'tear' is an action of ripping, shredding, or physically pulling something apart",
            "tair",
            "In this sentence, the word 'tear' means eyes watering, filling with tears, or about to cry",
            "teer")
    # Noun: "through the tear" = rift/portal
    if dep == "pobj" and head == "through":
        return "tair", "pos-pobj-through"
    return nli_decide(nli, sentence, "tear",
        "In this sentence, 'tear' refers to a rip, hole, cut, wound, injury, or physical damage in a material or flesh, or a rift, gap, tunnel, throughway, or transitional opening in a surface or barrier",
        "tair",
        "In this sentence, 'tear' refers to a teardrop, teardrop shape, liquid from the eye, or an emotional response",
        "teer")


def decide_polish(tag, dep, head, has_det, nli, sentence):
    # Capital P → proper noun (Polish nationality)
    word_in_sent = next((w for w in sentence.split() if w.lower() == "polish"), None)
    if word_in_sent and word_in_sent[0].isupper() and tag == "NNP":
        return "pole-ish", "cap-proper"
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "pollish", "pos-verb"
    if tag in ADJ_TAGS:
        return "pole-ish", "pos-adj"
    return "pollish", "pos-default"


# ── NLI-required decisions ────────────────────────────────────────────────────

BASS_MUSIC_HEADS = {"case", "player", "guitar", "drum", "line", "clef", "man", "solo", "note"}
BASS_FISH_HEADS  = {"boat", "fishing", "tournament", "lure", "angling", "farming"}
BASS_FISH_VERBS  = {"caught", "catch", "fry", "frying", "fried", "cook", "cooking", "cooked",
                    "ate", "eat", "eating", "hook", "hooked", "reel", "reeled", "land", "landed"}

_POSITIONAL_ROW = re.compile(r'\b(front|back|first|last|middle)\s+row\b', re.IGNORECASE)
_ROW_OF         = re.compile(r'\brow\s+of\b', re.IGNORECASE)


def decide_row(tag, dep, head, has_det, nli, sentence):
    # Compound nouns and positional uses are always the line/sequence sense
    if dep == "compound":
        return "ro", "pos-compound"
    if dep == "npadvmod":
        return "ro", "pos-npadvmod"
    if tag in VERB_PRESENT or tag in VERB_PAST:
        return "ro", "pos-verb"
    if _POSITIONAL_ROW.search(sentence):
        return "ro", "pos-positional"
    if _ROW_OF.search(sentence):
        return "ro", "pos-row-of"
    return nli_decide(nli, sentence, "row",
        "In this sentence, 'row' refers to a line, sequence, or series of things arranged in order, or the phrase 'in a row' meaning one after another consecutively",
        "ro",
        "In this sentence, 'row' is a loud verbal argument, quarrel, or shouting match between people",
        "rau")


def decide_bass(tag, dep, head, has_det, nli, sentence):
    if dep == "compound":
        if head in BASS_MUSIC_HEADS:
            return "base", "pos-compound-music"
        if head in BASS_FISH_HEADS:
            return "bass", "pos-compound-fish"
    if head in BASS_FISH_VERBS:
        return "bass", "pos-verb-fish"
    # Default to music (base) unless NLI is clearly confident it's fish
    q_music = "In this sentence, 'bass' refers to low-pitched sound, bass tones in music, a bass instrument (guitar, drum, etc.), or a bass musician"
    q_fish  = "In this sentence, 'bass' refers to a species of fish (largemouth bass, sea bass, striped bass, etc.) caught in fishing"
    result = nli(sentence, candidate_labels=[q_music, q_fish], multi_label=False)
    scores = dict(zip(result["labels"], result["scores"]))
    if scores[q_fish] > 0.70:
        return "bass", f"nli=bass({scores[q_fish]:.2f})"
    return "base", f"nli=base({scores[q_music]:.2f})"


def decide_bowed(tag, dep, head, has_det, nli, sentence):
    """bowed: physical deformation → boed; gesture → boughed"""
    if tag == "VBN" and not re.search(r'\bheads?\b', sentence, re.IGNORECASE):
        return "boed", "pos-vbn-deformation"
    return nli_decide(nli, sentence, "bowed",
        "In this sentence, a physical object or structure (shelf, fence, roof, wood, limb) has been bent or curved by weight or force",
        "boed",
        "In this sentence, a person deliberately lowered their head or body as a gesture of respect or greeting",
        "boughed")


def decide_wound(tag, dep, head, has_det, nli, sentence):
    """wound: noun → woond; verb (injure) → woond; verb (wind) → wow'nd"""
    if tag not in VERB_PAST and tag not in VERB_PRESENT:
        return "woond", "pos-noun"
    if tag == "VB":
        return "woond", "pos-vb-injure"
    return "wow'nd", "pos-verb-wind"


def decide_recreation(tag, dep, head, has_det, nli, sentence):
    """recreation: leisure activity → rek-reation; act of recreating → re-kreation"""
    if dep == "compound":
        return "rek-reation", "pos-compound-leisure"
    return nli_decide(nli, sentence, "recreation",
        "In this sentence, 'recreation' means the act of recreating or rebuilding something that previously existed",
        "re-kreation",
        "In this sentence, 'recreation' means a hobby, pastime, leisure activity, sport, or something done for enjoyment and relaxation",
        "rek-reation")


def decide_jesus(tag, dep, head, has_det, nli, sentence):
    """Jesus: biblical figure → Jesus; Spanish name → heysous"""
    return nli_decide(nli, sentence, "jesus",
        "In this sentence, 'Jesus' refers to the biblical Christian figure or is used as a religious exclamation",
        "Jesus",
        "In this sentence, 'Jesus' is being used as a Latin American or Spanish personal name (Jesús)",
        "heysous")


DECIDERS = {
    "record":     decide_record,
    "close":      decide_close,
    "read":       decide_read,
    "live":       decide_live,
    "object":     decide_object,
    "present":    decide_present,
    "sow":        decide_sow,
    "resume":     decide_resume,
    "refuse":     decide_refuse,
    "elaborate":  decide_elaborate,
    "estimate":   decide_estimate,
    "wind":       decide_wind,
    "invalid":    decide_invalid,
    "tear":       decide_tear,
    "polish":     decide_polish,
    "row":        decide_row,
    "bass":       decide_bass,
    "bowed":      decide_bowed,
    "wound":     decide_wound,
    "recreation": decide_recreation,
    "jesus":     decide_jesus,
}

# ── Decider Registry ──────────────────────────────────────────────────────────────
# Maps homograph word to its decider function.
# NLI_WORDS: deciders that require RoBERTa NLI for disambiguation.

NLI_WORDS = {"row", "bass", "bowed", "recreation", "jesus", "tear"}

# ── Test Runners ────────────────────────────────────────────────────────────────

def load_sentences(input_path, word, rules=None):
    """Parse input file, extract sentences containing target word."""
    sentences = []
    num = 0
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if rules:
            line = apply_replacements(line, rules)
        for sent in split_sentences(line):
            if re.search(rf'\b{re.escape(word)}\b', sent, re.IGNORECASE):
                num += 1
                sentences.append((num, sent))
    return sentences
    """Run analysis on one homograph — write results to *_hybrid_results.txt."""
    if output_folder is None:
        output_folder = OUT_DIR

    sentences = load_sentences(input_path, word, rules)
    if not sentences:
        print(f"  No sentences found for '{word}'")
        return

    print(f"\n--- {word} ({len(sentences)} sentences) ---")
    decider = DECIDERS[word]
    results = []
    pred_dist = {}

    for num, text in sentences:
        token_infos = get_token_info(nlp, text, word)
        for idx, (tag, dep, head, has_det) in enumerate(token_infos):
            pred, route = decider(tag, dep, head, has_det, nli, text)
            pred_dist[pred] = pred_dist.get(pred, 0) + 1
            inst_label = f"{num}.{idx+1}" if len(token_infos) > 1 else str(num)
            results.append((inst_label, text, pred, tag, dep, head, route))

    out_path = output_folder / f"{word}_hybrid_results.txt"
    lines = []
    lines.append(f"HYBRID TEST: {word.upper()}")
    lines.append(f"Sentences: {len(sentences)}")
    lines.append(f"Prediction distribution: {dict(sorted(pred_dist.items()))}")
    lines.append("=" * 90)
    lines.append("")
    for num, text, pred, tag, dep, head, route in results:
        lines.append(f"[{num}] {text}")
        lines.append(f"  pred={pred}  tag={tag}  dep={dep}  head={head}  route={route}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Results written to {out_path.name}")
    print(f"  Prediction distribution: {dict(sorted(pred_dist.items()))}")


def run_word_collect(word, input_path, nlp, nli, rules):
    """Collect results into list for combining — returns list of dicts."""
    """Like run_word() but returns results list instead of writing file."""
    sentences = load_sentences(input_path, word, rules)
    if not sentences:
        return []

    results = []
    pred_dist = {}
    decider = DECIDERS[word]

    for num, text in sentences:
        token_infos = get_token_info(nlp, text, word)
        for idx, (tag, dep, head, has_det) in enumerate(token_infos):
            pred, route = decider(tag, dep, head, has_det, nli, text)
            pred_dist[pred] = pred_dist.get(pred, 0) + 1
            inst_label = f"{num}.{idx+1}" if len(token_infos) > 1 else str(num)
            results.append({
                "word": word,
                "num": inst_label,
                "text": text,
                "pred": pred,
                "tag": tag,
                "dep": dep,
            "head": head,
            "route": route
        })

    return results


def run_word(word, input_path, nlp, nli, rules, output_folder=None):
    """Process single homograph and write results to individual file."""
    if output_folder is None:
        output_folder = Path(".")

    sentences = load_sentences(input_path, word, rules)
    if not sentences:
        print(f"No sentences found for '{word}'")
        return

    lines = []
    lines.append(f"HOMOGRAPH: {word}")
    lines.append(f"Input file: {input_path.name}")
    lines.append(f"Total sentences: {len(sentences)}")
    lines.append("=" * 60)
    lines.append("")

    pred_dist = {}
    decider = DECIDERS[word]

    for num, text in sentences:
        token_infos = get_token_info(nlp, text, word)
        for idx, (tag, dep, head, has_det) in enumerate(token_infos):
            pred, route = decider(tag, dep, head, has_det, nli, text)
            pred_dist[pred] = pred_dist.get(pred, 0) + 1
            inst_label = f"{num}.{idx+1}" if len(token_infos) > 1 else str(num)
            lines.append(f"[{inst_label}] {text}")
            lines.append(f"  pred={pred}  tag={tag}  dep={dep}  head={head}  route={route}")
            lines.append("")

    lines.append("=" * 60)
    lines.append(f"Prediction distribution: {dict(sorted(pred_dist.items()))}")

    out_path = output_folder / f"{word}_results.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Results for '{word}' written to {out_path.name}")


# ── Combined Output ────────────────────────────────────────────────────────────────

def write_combined_results(results, input_path, output_folder=None):
    """Write all homograph results to single combined file."""
    if output_folder is None:
        output_folder = OUT_DIR

    out_path = output_folder / f"{input_path.stem}_hybrid_results.txt"
    lines = []
    lines.append("HYBRID TEST: ALL HOMOGRAPHS")
    lines.append(f"Total sentences: {len(results)}")
    lines.append("=" * 90)
    lines.append("")

    by_word = {}
    for r in results:
        word = r["word"]
        if word not in by_word:
            by_word[word] = {"total": 0, "preds": {}}
        by_word[word]["total"] += 1
        pred = r["pred"]
        by_word[word]["preds"][pred] = by_word[word]["preds"].get(pred, 0) + 1

    for word in sorted(by_word.keys()):
        stats = by_word[word]
        lines.append(f"{word.upper()}: {stats['total']} sentences, predictions: {dict(sorted(stats['preds'].items()))}")

    lines.append("")
    lines.append("")

    for word in sorted(by_word.keys()):
        word_results = [r for r in results if r["word"] == word]
        lines.append(f"--- {word.upper()} ---")
        lines.append("")
        for r in word_results:
            lines.append(f"[{r['num']}] {r['text']}")
            lines.append(f"  pred={r['pred']}  tag={r['tag']}  dep={r['dep']}  head={r['head']}  route={r['route']}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Combined results written to {out_path.name}")


# ── Production Mode ────────────────────────────────────────────────────────────────

def run_production(input_path, nlp, nli, rules, output_folder=None):
    """Process full text file, replacing all homograph occurrences."""
    if output_folder is None:
        output_folder = input_path.parent

    output_path = output_folder / f"{input_path.stem}_processed{input_path.suffix}"

    text = input_path.read_text(encoding="utf-8")
    lines = text.split('\n')

    output_lines = []
    for line in lines:
        if not line.strip():
            output_lines.append(line)
            continue

        if rules:
            line = apply_replacements(line, rules)

        sentences = split_sentences(line)
        modified_sentences = []

        for sent in sentences:
            modified_sent = sent
            for word in DECIDERS:
                if re.search(rf'\b{re.escape(word)}\b', modified_sent, re.IGNORECASE):
                    token_infos = get_token_info(nlp, modified_sent, word)
                    decider = DECIDERS[word]
                    replacements = []
                    for tag, dep, head, has_det in token_infos:
                        pred, route = decider(tag, dep, head, has_det, nli, modified_sent)
                        replacements.append(pred)
                    for pred in replacements:
                        modified_sent = re.sub(
                            r'\b' + re.escape(word) + r'\b',
                            pred,
                            modified_sent,
                            flags=re.IGNORECASE,
                            count=1
                        )
            modified_sentences.append(modified_sent)

        output_lines.append(' '.join(modified_sentences))

    output_path.write_text('\n'.join(output_lines), encoding="utf-8")
    print(f"Production output written to {output_path}")


# ── CLI Entry Point ───────────────────────────────────────────────────────────────────

def main():
    """Interactive CLI: file selection, mode selection, model loading, run analysis/production."""
    script_dir = Path(__file__).parent
    input_folder = script_dir / "InputText"
    output_folder = script_dir / "OutputText"

    output_folder.mkdir(exist_ok=True)

    if not input_folder.exists():
        print(f"Error: InputText folder not found at {input_folder}")
        sys.exit(1)

    input_files = sorted([f for f in input_folder.iterdir() if f.is_file() and not f.name.startswith('.')])

    if not input_files:
        print(f"No files found in {input_folder}")
        sys.exit(1)

    print("\nAvailable files:")
    for i, f in enumerate(input_files, 1):
        print(f"  {i}) {f.name}")

    choice = input("\nSelect file [1]: ").strip() or "1"
    try:
        file_idx = int(choice) - 1
        if file_idx < 0 or file_idx >= len(input_files):
            print(f"Invalid choice")
            sys.exit(1)
        input_path = input_files[file_idx]
    except ValueError:
        print("Invalid input")
        sys.exit(1)

    print(f"\nSelected: {input_path.name}")

    print("\nMode:")
    print("  1) Test/analysis  (predictions + route info)")
    print("  2) Production     (replace homographs in text, save output)")
    mode_choice = input("Choice [1]: ").strip() or "1"

    rules = load_replacements()
    if rules:
        print(f"Loaded {len(rules)} replacement rules from {REPLACE_FILE.name}")

    print(f"Loading {POS_MODEL}...")
    spacy.prefer_gpu()
    nlp = spacy.load(POS_MODEL)

    if mode_choice == "2":
        print(f"Loading {NLI_MODEL}...")
        device = 0 if torch.cuda.is_available() else -1
        nli = hf_pipeline("zero-shot-classification", model=NLI_MODEL, device=device)
        run_production(input_path, nlp, nli, rules, output_folder)
    else:
        words = list(DECIDERS.keys())
        nli = None
        if any(w in NLI_WORDS for w in words):
            print(f"Loading {NLI_MODEL}...")
            device = 0 if torch.cuda.is_available() else -1
            nli = hf_pipeline("zero-shot-classification", model=NLI_MODEL, device=device)

        print("\nOutput:")
        print("  a) One combined file")
        print("  b) One file per homograph")
        output_choice = input("Choice [b]: ").strip().lower() or "b"

        if output_choice == "a":
            results_list = []
            for word in words:
                results = run_word_collect(word, input_path, nlp, nli, rules)
                results_list.extend(results)
            write_combined_results(results_list, input_path, output_folder)
        else:
            for word in words:
                run_word(word, input_path, nlp, nli, rules, output_folder)


if __name__ == "__main__":
    main()
