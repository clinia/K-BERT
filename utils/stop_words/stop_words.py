class StopWords:
    """
    This class holds the french and english stopwords determinants that are of length less or equal to three.
    They are meant to be used to keep the secondary tagger from tagging entities with those determinants at the
    beginning or the end of the segment.
    """

    def __init__(self) -> None:

        # Unwanted characters that may allow matching if used in the beginning or the end of a segment
        self.stop_char = {
            "-",
            "_",
            "/",
            "\\",
            "+",
            "=",
            "@",
            "#",
            "$",
            "%",
            "&",
            "*",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            "~",
            "^",
            "°",
            ",",
            ".",
            "!",
            "?",
        }

        self.en = {
            "her",
            "was",
            "up",
            "get",
            "six",
            "had",
            "'d",
            "in",
            "re",
            "a",
            "we",
            "via",
            "can",
            "at",
            "not",
            "for",
            "n't",
            "me",
            "any",
            "out",
            "an",
            "she",
            "by",
            "'re",
            "now",
            "to",
            "few",
            "all",
            "say",
            "or",
            "own",
            "'m",
            "our",
            "him",
            # "‘s",
            "be",
            "'ll",
            "its",
            "if",
            "he",
            "my",
            "per",
            "are",
            "you",
            "'m",
            "yet",
            # "’s",
            "see",
            "am",
            "'ll",
            "who",
            "put",
            "ca",
            "'ll",
            "one",
            "did",
            "may",
            "the",
            "but",
            "is",
            "go",
            "i",
            "too",
            "so",
            "on",
            "it",
            "'m",
            "'d",
            "n't",
            "and",
            # "'s",
            "us",
            "ten",
            "no",
            "two",
            "how",
            "due",
            "his",
            "'ve",
            "nor",
            "has",
            "off",
            "'d",
            "as",
            "top",
            "of",
            "'ve",
            "do",
            "why",
        }

        # Add stop char
        self.en.update(self.stop_char)

        self.fr = {
            "si",
            "n'",
            "euh",
            "est",
            "m'",
            "hem",
            "pu",
            "l'",
            "ohé",
            "en",
            "d'",
            "là",
            "las",
            "vif",
            "n'",
            "six",
            "ils",
            "nos",
            "vu",
            "par",
            "qu'",
            "aie",
            "dix",
            "s'",
            "a",
            "hum",
            "moi",
            "via",
            "ouf",
            "hé",
            "ai",
            "tic",
            "ça",
            "ni",
            "je",
            "qui",
            "bah",
            "ait",
            "me",
            "eux",
            "tac",
            "tel",
            "ont",
            "un",
            "dit",
            "les",
            "fi",
            "paf",
            "des",
            "dès",
            "vos",
            "pif",
            "hi",
            "nul",
            "na",
            "ci",
            "ô",
            "ta",
            "uns",
            "j'",
            "tu",
            "sur",
            "pas",
            "té",
            "â",
            "de",
            "à",
            "t'",
            "hep",
            "il",
            "ne",
            "soi",
            "j'",
            "ce",
            "pur",
            "une",
            "zut",
            "s'",
            "ore",
            "et",
            "bat",
            "etc",
            "vé",
            "non",
            "se",
            "du",
            "sa",
            "été",
            "hou",
            "mes",
            "au",
            "o",
            "cet",
            "va",
            "bas",
            "que",
            "tes",
            "eu",
            "la",
            "lui",
            "t'",
            "l'",
            "hop",
            "pan",
            "ou",
            "c'",
            "ho",
            "m'",
            "d'",
            "ses",
            "son",
            "i",
            "le",
            "ces",
            "qu'",
            "pff",
            "vas",
            "hue",
            "aux",
            "on",
            "mon",
            "ès",
            "peu",
            "ton",
            "ah",
            "eh",
            "toc",
            "da",
            "olé",
            "ma",
            "te",
            "où",
            "car",
            "es",
            "toi",
            "oh",
            "as",
            "c'",
            "lès",
            "ha",
            "hui",
        }

        # Add stop char
        self.fr.update(self.stop_char)
