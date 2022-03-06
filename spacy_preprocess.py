# Re-using the blueprint from Chapter 4 but adapting to add additional steps specific to this dataset
import re  # ##

import spacy  # ##
import textacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex, compile_prefix_regex, compile_suffix_regex
from textacy.preprocessing.replace import urls as replace_urls


def custom_tokenizer(nlp):

    # use default patterns except the ones matched by re.search
    prefixes = [
        pattern for pattern in nlp.Defaults.prefixes if pattern not in ["-", "_", "#"]
    ]
    suffixes = [pattern for pattern in nlp.Defaults.suffixes if pattern not in ["_"]]
    infixes = [
        pattern for pattern in nlp.Defaults.infixes if not re.search(pattern, "xx-xx")
    ]

    return Tokenizer(
        vocab=nlp.vocab,
        rules=nlp.Defaults.tokenizer_exceptions,
        prefix_search=compile_prefix_regex(prefixes).search,
        suffix_search=compile_suffix_regex(suffixes).search,
        infix_finditer=compile_infix_regex(infixes).finditer,
        token_match=nlp.Defaults.token_match,
    )


nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = custom_tokenizer(nlp)


def extract_lemmas(doc, **kwargs):
    return [t.lemma_ for t in textacy.extract.words(doc, **kwargs)]


def extract_noun_chunks(doc, include_pos=["NOUN"], sep="_"):
    chunks = []
    for noun_chunk in doc.noun_chunks:
        chunk = [token.lemma_ for token in noun_chunk if token.pos_ in include_pos]
        if len(chunk) >= 2:
            chunks.append(sep.join(chunk))
    return chunks


def extract_entities(doc, include_types=None, sep="_"):

    ents = textacy.extract.entities(
        doc,
        include_types=include_types,
        exclude_types=None,
        drop_determiners=True,
        min_freq=1,
    )

    return [re.sub("\s+", sep, e.lemma_) + "/" + e.label_ for e in ents]


def spacy_clean(text):
    # Replace URLs
    text = replace_urls(text)

    # Replace semi-colons (relevant in Java code ending)
    text = text.replace(";", "")

    # Replace character tabs (present as literal in description field)
    text = text.replace("\t", "")

    # Find and remove any stack traces - doesn't fix all code fragments but removes many exceptions
    start_loc = text.find("Stack trace:")
    text = text[:start_loc]

    # Remove Hex Code
    text = re.sub(r"(\w+)0x\w+", "", text)

    # Initialize Spacy
    doc = nlp(text)

    # From Blueprint function
    lemmas = extract_lemmas(
        doc,
        exclude_pos=["PART", "PUNCT", "DET", "PRON", "SYM", "SPACE", "NUM"],
        filter_stops=True,
        filter_nums=True,
        filter_punct=True,
    )

    return lemmas