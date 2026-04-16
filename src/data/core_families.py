"""Core concept families for the LAP paper.

Five heterogeneous families used for within-concept layer analysis:
arithmetic, geography, sequence, word_transform, analogy.
"""

import random
import itertools
from src.data.base import ProbeDataset, ProbePrompt

class ArithmeticProbe(ProbeDataset):
    """Arithmetic via next-token prediction.

    Single-digit results only so answers are single tokens.
    Spaced format with trailing space: "2 + 5 = " so model predicts bare digit.

    Easy: a+b where a+b<=9
    Medium: a*b where a*b<=9
    Hard: a+b+c where a+b+c<=9
    """

    family_name = "arithmetic"

    def __init__(self, n_prompts: int = 500, seed: int = 42):
        self.n_prompts = n_prompts
        self.seed = seed

    def _make_prompt(self, expression: str, correct: int, rng, difficulty: str,
                     **extra_meta) -> ProbePrompt:
        distractors = set()
        while len(distractors) < 3:
            d = rng.randint(0, 9)
            if d != correct:
                distractors.add(d)
        candidates = [str(correct)] + [str(d) for d in distractors]
        rng.shuffle(candidates)
        correct_idx = candidates.index(str(correct))
        return ProbePrompt(
            prompt_text=f"{expression} = ",
            candidates=candidates,
            correct_index=correct_idx,
            family=self.family_name,
            metadata={"difficulty": difficulty, "expression": expression, **extra_meta},
        )

    def load(self) -> list[ProbePrompt]:
        rng = random.Random(self.seed)
        prompts = []
        seen = set()

        def add(expr, correct, difficulty, **meta):
            if expr not in seen and len(prompts) < self.n_prompts:
                seen.add(expr)
                prompts.append(self._make_prompt(expr, correct, rng, difficulty, **meta))

        # Single-digit a + b (36 problems)
        for a in range(1, 10):
            for b in range(1, 10):
                if a + b <= 9:
                    add(f"{a} + {b}", a + b, "easy")

        # Single-digit a * b, skip trivial (6 problems)
        for a in range(2, 10):
            for b in range(2, 10):
                if a * b <= 9:
                    add(f"{a} * {b}", a * b, "easy")

        # Subtraction: a - b where 0 <= result <= 9 (many problems)
        for a in range(1, 20):
            for b in range(1, a):
                if a - b <= 9:
                    add(f"{a} - {b}", a - b, "medium")

        # Two-digit + single-digit with single-digit result: (10+a) - b
        for a in range(0, 10):
            for b in range(1, 20):
                result = (10 + a) - b
                if 0 <= result <= 9 and b != 10 + a:
                    add(f"{10 + a} - {b}", result, "medium")

        # Three-operand: a + b + c <= 9
        for a in range(1, 5):
            for b in range(1, 5):
                for c in range(1, 5):
                    if a + b + c <= 9:
                        add(f"{a} + {b} + {c}", a + b + c, "hard")

        # a + b - c with single-digit result
        for a in range(1, 10):
            for b in range(1, 10):
                for c in range(1, a + b):
                    result = a + b - c
                    if 0 <= result <= 9 and c < a + b:
                        add(f"{a} + {b} - {c}", result, "hard")

        rng.shuffle(prompts)
        return prompts[:self.n_prompts]

class GeographyProbe(ProbeDataset):
    """Geography knowledge: capitals, continents, languages, regions.

    Multiple relation types, all with unambiguous single-token answers.
    """

    family_name = "geography"

    # (capital, country)
    _CAPITALS = [
        ("London", "England"), ("Paris", "France"), ("Tokyo", "Japan"),
        ("Berlin", "Germany"), ("Madrid", "Spain"), ("Rome", "Italy"),
        ("Moscow", "Russia"), ("Beijing", "China"), ("Seoul", "Korea"),
        ("Bangkok", "Thailand"), ("Athens", "Greece"), ("Warsaw", "Poland"),
        ("Stockholm", "Sweden"), ("Oslo", "Norway"), ("Helsinki", "Finland"),
        ("Copenhagen", "Denmark"), ("Amsterdam", "Netherlands"),
        ("Brussels", "Belgium"), ("Vienna", "Austria"), ("Prague", "Czechia"),
        ("Budapest", "Hungary"), ("Bucharest", "Romania"), ("Sofia", "Bulgaria"),
        ("Zagreb", "Croatia"), ("Belgrade", "Serbia"), ("Lisbon", "Portugal"),
        ("Dublin", "Ireland"), ("Ankara", "Turkey"), ("Cairo", "Egypt"),
        ("Tehran", "Iran"), ("Baghdad", "Iraq"),
        ("Jakarta", "Indonesia"), ("Manila", "Philippines"),
        ("Hanoi", "Vietnam"), ("Taipei", "Taiwan"),
        ("Canberra", "Australia"),
        ("Ottawa", "Canada"), ("Havana", "Cuba"), ("Lima", "Peru"),
        ("Santiago", "Chile"), ("Bogota", "Colombia"),
        ("Brasilia", "Brazil"), ("Nairobi", "Kenya"), ("Accra", "Ghana"),
        ("Dakar", "Senegal"), ("Rabat", "Morocco"),
        ("Kyiv", "Ukraine"), ("Minsk", "Belarus"),
    ]

    # (country, continent)
    _CONTINENTS = [
        ("France", "Europe"), ("Germany", "Europe"), ("Spain", "Europe"),
        ("Italy", "Europe"), ("Poland", "Europe"), ("Sweden", "Europe"),
        ("Greece", "Europe"), ("Portugal", "Europe"), ("Ireland", "Europe"),
        ("Norway", "Europe"), ("Finland", "Europe"), ("Denmark", "Europe"),
        ("Japan", "Asia"), ("China", "Asia"), ("Korea", "Asia"),
        ("Thailand", "Asia"), ("Vietnam", "Asia"), ("Indonesia", "Asia"),
        ("India", "Asia"), ("Iran", "Asia"), ("Iraq", "Asia"),
        ("Turkey", "Asia"), ("Philippines", "Asia"), ("Taiwan", "Asia"),
        ("Egypt", "Africa"), ("Kenya", "Africa"), ("Ghana", "Africa"),
        ("Nigeria", "Africa"), ("Senegal", "Africa"), ("Morocco", "Africa"),
        ("Ethiopia", "Africa"), ("Tanzania", "Africa"), ("Uganda", "Africa"),
        ("Brazil", "America"), ("Canada", "America"), ("Cuba", "America"),
        ("Mexico", "America"), ("Peru", "America"), ("Chile", "America"),
        ("Colombia", "America"), ("Argentina", "America"),
        ("Australia", "Oceania"),
    ]

    # (country, language)
    _LANGUAGES = [
        ("France", "French"), ("Germany", "German"), ("Spain", "Spanish"),
        ("Italy", "Italian"), ("Portugal", "Portuguese"), ("Russia", "Russian"),
        ("China", "Chinese"), ("Japan", "Japanese"), ("Brazil", "Portuguese"),
        ("Turkey", "Turkish"), ("Greece", "Greek"), ("Poland", "Polish"),
        ("Sweden", "Swedish"), ("Korea", "Korean"), ("Thailand", "Thai"),
        ("Netherlands", "Dutch"), ("Denmark", "Danish"), ("Finland", "Finnish"),
        ("Norway", "Norwegian"), ("Romania", "Romanian"), ("Hungary", "Hungarian"),
        ("Vietnam", "Vietnamese"), ("Indonesia", "Indonesian"),
        ("Israel", "Hebrew"), ("Egypt", "Arabic"), ("Iran", "Persian"),
        ("Ukraine", "Ukrainian"),
        ("Croatia", "Croatian"), ("Serbia", "Serbian"),
        ("Bulgaria", "Bulgarian"),
        ("Mexico", "Spanish"), ("Argentina", "Spanish"),
        ("Colombia", "Spanish"), ("Chile", "Spanish"),
        ("Peru", "Spanish"), ("Cuba", "Spanish"),
        ("Canada", "English"), ("Australia", "English"),
        ("Ireland", "English"),
    ]

    # (country, currency)
    _CURRENCIES = [
        ("Japan", "yen"), ("Britain", "pound"), ("America", "dollar"),
        ("Europe", "euro"), ("India", "rupee"), ("China", "yuan"),
        ("Russia", "ruble"), ("Korea", "won"), ("Brazil", "real"),
        ("Mexico", "peso"), ("Thailand", "baht"), ("Turkey", "lira"),
        ("Sweden", "krona"), ("Switzerland", "franc"), ("Poland", "zloty"),
    ]

    _CAPITAL_TEMPLATES = [
        "{capital} is the capital of",
        "The capital city {capital} is located in",
    ]

    _CONTINENT_TEMPLATES = [
        "{country} is a country in",
        "{country} is located in",
        "The continent of {country} is",
    ]

    _LANGUAGE_TEMPLATES = [
        "In {country}, people speak",
        "The official language of {country} is",
        "People from {country} speak",
        "{country}'s national language is",
        "The language spoken in {country} is",
        "Citizens of {country} typically speak",
        "In {country}, the main language is",
        "The primary language of {country} is",
    ]

    _CURRENCY_TEMPLATES = [
        "The currency of {country} is the",
        "In {country}, they pay with the",
    ]

    def __init__(self, n_prompts: int = 500):
        self.n_prompts = n_prompts

    def load(self) -> list[ProbePrompt]:
        rng = random.Random(42)
        prompts = []

        # Capitals
        for capital, country in self._CAPITALS:
            for template in self._CAPITAL_TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(capital=capital),
                    candidates=[country],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"relation": "capital", "capital": capital, "country": country},
                ))

        # Continents
        for country, continent in self._CONTINENTS:
            for template in self._CONTINENT_TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(country=country),
                    candidates=[continent],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"relation": "continent", "country": country, "continent": continent},
                ))

        # Languages
        for country, language in self._LANGUAGES:
            for template in self._LANGUAGE_TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(country=country),
                    candidates=[language],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"relation": "language", "country": country, "language": language},
                ))

        # Currencies
        for country, currency in self._CURRENCIES:
            for template in self._CURRENCY_TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(country=country),
                    candidates=[currency],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"relation": "currency", "country": country, "currency": currency},
                ))

        rng.shuffle(prompts)
        return prompts[:self.n_prompts]


class AnalogyProbe(ProbeDataset):
    """Analogy/relationship completion.

    Prompt: "hot is to cold as big is to"
    Correct next token: "small"
    """

    family_name = "analogy"

    _ANALOGIES = [
        # Opposites
        ("hot", "cold", "big", "small"), ("up", "down", "left", "right"),
        ("day", "night", "summer", "winter"), ("fast", "slow", "tall", "short"),
        ("open", "closed", "on", "off"), ("happy", "sad", "rich", "poor"),
        ("light", "dark", "loud", "quiet"), ("hard", "soft", "rough", "smooth"),
        ("wet", "dry", "hot", "cold"), ("full", "empty", "thick", "thin"),
        ("strong", "weak", "brave", "cowardly"), ("love", "hate", "peace", "war"),
        ("begin", "end", "start", "finish"), ("buy", "sell", "give", "take"),
        ("push", "pull", "come", "go"), ("win", "lose", "pass", "fail"),
        ("sweet", "bitter", "kind", "cruel"), ("alive", "dead", "awake", "asleep"),
        ("safe", "dangerous", "clean", "dirty"), ("early", "late", "first", "last"),
        ("bright", "dim", "sharp", "dull"), ("deep", "shallow", "wide", "narrow"),
        ("true", "false", "real", "fake"), ("above", "below", "over", "under"),
        ("friend", "enemy", "ally", "rival"),
        # Family relations
        ("king", "queen", "prince", "princess"), ("father", "mother", "son", "daughter"),
        ("brother", "sister", "uncle", "aunt"), ("man", "woman", "boy", "girl"),
        # Animals
        ("dog", "puppy", "cat", "kitten"), ("cow", "calf", "horse", "foal"),
        # Directions
        ("north", "south", "east", "west"),
        # Time
        ("morning", "evening", "dawn", "dusk"),
        ("floor", "ceiling", "bottom", "top"),
        # More opposites
        ("black", "white", "dark", "light"),
        ("long", "short", "wide", "narrow"),
        ("high", "low", "deep", "shallow"),
        ("good", "bad", "right", "wrong"),
        ("young", "old", "new", "ancient"),
        ("near", "far", "close", "distant"),
        ("front", "back", "head", "tail"),
        ("rise", "fall", "grow", "shrink"),
        ("enter", "exit", "arrive", "depart"),
        ("laugh", "cry", "smile", "frown"),
        ("speak", "listen", "write", "read"),
        ("create", "destroy", "build", "demolish"),
        ("accept", "reject", "include", "exclude"),
        ("remember", "forget", "find", "lose"),
        ("attack", "defend", "offense", "defense"),
        ("lead", "follow", "teach", "learn"),
        ("lend", "borrow", "export", "import"),
        # Size/quantity
        ("giant", "tiny", "huge", "small"),
        ("many", "few", "more", "less"),
        ("increase", "decrease", "expand", "contract"),
        ("add", "subtract", "multiply", "divide"),
        # Temperature/weather
        ("summer", "winter", "warm", "cold"),
        ("sun", "moon", "day", "night"),
        ("rain", "drought", "flood", "desert"),
        # Social/power
        ("master", "servant", "employer", "employee"),
        ("rich", "poor", "wealth", "poverty"),
        ("freedom", "slavery", "liberty", "captivity"),
        ("victory", "defeat", "success", "failure"),
        # Movement
        ("advance", "retreat", "forward", "backward"),
        ("ascend", "descend", "climb", "fall"),
        ("expand", "contract", "grow", "shrink"),
        ("inhale", "exhale", "breathe", "release"),
        # Senses
        ("loud", "silent", "bright", "dark"),
        ("sweet", "sour", "smooth", "rough"),
        # More family/gender
        ("husband", "wife", "groom", "bride"),
        ("nephew", "niece", "grandson", "granddaughter"),
        ("hero", "heroine", "actor", "actress"),
        ("waiter", "waitress", "host", "hostess"),
        # Geography analog
        ("mountain", "valley", "peak", "basin"),
        ("island", "mainland", "ocean", "continent"),
        ("city", "village", "urban", "rural"),
        # Material states
        ("solid", "liquid", "ice", "water"),
        ("freeze", "melt", "harden", "soften"),
        # Time
        ("past", "future", "yesterday", "tomorrow"),
        ("ancient", "modern", "old", "new"),
        ("begin", "finish", "dawn", "dusk"),
        ("infant", "adult", "child", "parent"),
        # Academic
        ("question", "answer", "problem", "solution"),
        ("student", "teacher", "patient", "doctor"),
        ("theory", "practice", "abstract", "concrete"),
        ("cause", "effect", "action", "reaction"),
        # Food
        ("hungry", "full", "thirsty", "satisfied"),
        ("raw", "cooked", "fresh", "stale"),
    ]

    def __init__(self, n_prompts: int = 500):
        self.n_prompts = n_prompts

    def load(self) -> list[ProbePrompt]:
        rng = random.Random(42)
        prompts = []

        for a, b, c, d in self._ANALOGIES:
            # A is to B as C is to D
            prompts.append(ProbePrompt(
                prompt_text=f"{a} is to {b} as {c} is to",
                candidates=[d],
                correct_index=0,
                family=self.family_name,
                metadata={"a": a, "b": b, "c": c, "d": d},
            ))
            # B is to A as D is to C
            prompts.append(ProbePrompt(
                prompt_text=f"{b} is to {a} as {d} is to",
                candidates=[c],
                correct_index=0,
                family=self.family_name,
                metadata={"a": b, "b": a, "c": d, "d": c},
            ))
            # C is to D as A is to B
            prompts.append(ProbePrompt(
                prompt_text=f"{c} is to {d} as {a} is to",
                candidates=[b],
                correct_index=0,
                family=self.family_name,
                metadata={"a": c, "b": d, "c": a, "d": b},
            ))
            # D is to C as B is to A
            prompts.append(ProbePrompt(
                prompt_text=f"{d} is to {c} as {b} is to",
                candidates=[a],
                correct_index=0,
                family=self.family_name,
                metadata={"a": d, "b": c, "c": b, "d": a},
            ))

        rng.shuffle(prompts)
        return prompts[:self.n_prompts]


class SequenceProbe(ProbeDataset):
    """Sequence completion: temporal, numerical, alphabetical.

    Prompt: "Monday, Tuesday, Wednesday,"
    Correct next token: "Thursday"
    """

    family_name = "sequence"

    def __init__(self, n_prompts: int = 500, seed: int = 42):
        self.n_prompts = n_prompts
        self.seed = seed

    def load(self) -> list[ProbePrompt]:
        rng = random.Random(self.seed)
        prompts = []

        # Days of the week — all starting positions, various lengths
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for length in [2, 3, 4, 5, 6]:
            for start in range(7):
                seq = [days[(start + i) % 7] for i in range(length)]
                answer = days[(start + length) % 7]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "days", "length": length},
                ))

        # Months — all starting positions, various lengths
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        for length in [2, 3, 4, 5, 6, 7]:
            for start in range(12):
                seq = [months[(start + i) % 12] for i in range(length)]
                answer = months[(start + length) % 12]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "months", "length": length},
                ))

        # Seasons
        seasons = ["spring", "summer", "autumn", "winter"]
        for length in [2, 3]:
            for start in range(4):
                seq = [seasons[(start + i) % 4] for i in range(length)]
                answer = seasons[(start + length) % 4]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "seasons", "length": length},
                ))

        # Ordinals
        ordinals = ["first", "second", "third", "fourth", "fifth",
                    "sixth", "seventh", "eighth", "ninth", "tenth"]
        for length in [2, 3, 4, 5, 6]:
            for start in range(len(ordinals) - length):
                seq = ordinals[start:start + length]
                answer = ordinals[start + length]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "ordinals", "length": length},
                ))

        # Number words
        number_words = ["one", "two", "three", "four", "five",
                        "six", "seven", "eight", "nine", "ten"]
        for length in [2, 3, 4, 5, 6]:
            for start in range(len(number_words) - length):
                seq = number_words[start:start + length]
                answer = number_words[start + length]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "number_words", "length": length},
                ))

        # Planets
        planets = ["Mercury", "Venus", "Earth", "Mars", "Jupiter",
                   "Saturn", "Uranus", "Neptune"]
        for length in [2, 3, 4, 5]:
            for start in range(len(planets) - length):
                seq = planets[start:start + length]
                answer = planets[start + length]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "planets", "length": length},
                ))

        # Phonetic alphabet
        phonetic = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
                    "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima", "Mike",
                    "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra",
                    "Tango", "Uniform", "Victor", "Whiskey"]
        for length in [2, 3, 4, 5]:
            for start in range(len(phonetic) - length):
                seq = phonetic[start:start + length]
                answer = phonetic[start + length]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "phonetic", "length": length},
                ))

        # Musical notes (solfege)
        solfege = ["do", "re", "mi", "fa", "sol", "la", "ti"]
        for length in [2, 3, 4, 5]:
            for start in range(len(solfege) - length):
                seq = solfege[start:start + length]
                answer = solfege[start + length]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "solfege", "length": length},
                ))

        # Alphabet sequences — various lengths and start positions
        letters = list("abcdefghijklmnopqrstuvwxyz")
        for length in [3, 4, 5, 6, 7]:
            for start in range(26 - length):
                seq = letters[start:start + length]
                answer = letters[start + length]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "alphabet", "length": length},
                ))

        # Counting sequences (single digit answers only)
        for start in range(1, 6):
            for length in [3, 4, 5]:
                if start + length <= 9:
                    seq = list(range(start, start + length))
                    answer = str(start + length)
                    prompts.append(ProbePrompt(
                        prompt_text=", ".join(str(x) for x in seq) + ",",
                        candidates=[answer],
                        correct_index=0,
                        family=self.family_name,
                        metadata={"type": "counting", "length": length},
                    ))

        # Even numbers: 2, 4, 6, -> 8
        for start in [2, 4]:
            for length in [2, 3]:
                seq = list(range(start, start + length * 2, 2))
                answer = str(start + length * 2)
                if int(answer) <= 9:
                    prompts.append(ProbePrompt(
                        prompt_text=", ".join(str(x) for x in seq) + ",",
                        candidates=[answer],
                        correct_index=0,
                        family=self.family_name,
                        metadata={"type": "even", "length": length},
                    ))

        # Roman numerals
        roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        for length in [2, 3, 4, 5]:
            for start in range(len(roman) - length):
                seq = roman[start:start + length]
                answer = roman[start + length]
                prompts.append(ProbePrompt(
                    prompt_text=", ".join(seq) + ",",
                    candidates=[answer],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"type": "roman", "length": length},
                ))

        rng.shuffle(prompts)
        return prompts[:self.n_prompts]
class WordTransformProbe(ProbeDataset):
    """Word transformations: opposite, plural, past tense.

    Scalable family — hundreds of entries per transform type.
    """

    family_name = "word_transform"

    # (word, opposite)
    _OPPOSITES = [
        ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("tall", "short"),
        ("happy", "sad"), ("light", "dark"), ("loud", "quiet"), ("hard", "soft"),
        ("wet", "dry"), ("full", "empty"), ("strong", "weak"), ("rich", "poor"),
        ("old", "young"), ("clean", "dirty"), ("safe", "dangerous"),
        ("open", "closed"), ("deep", "shallow"), ("wide", "narrow"),
        ("thick", "thin"), ("rough", "smooth"), ("sharp", "dull"),
        ("sweet", "bitter"), ("kind", "cruel"), ("brave", "cowardly"),
        ("alive", "dead"), ("awake", "asleep"), ("early", "late"),
        ("near", "far"), ("true", "false"), ("good", "bad"),
        ("high", "low"), ("long", "short"), ("new", "old"),
        ("right", "wrong"), ("black", "white"), ("bright", "dim"),
        ("cheap", "expensive"), ("easy", "difficult"), ("heavy", "light"),
        ("loose", "tight"), ("major", "minor"), ("positive", "negative"),
        ("private", "public"), ("rare", "common"), ("raw", "cooked"),
        ("real", "fake"), ("rude", "polite"), ("rural", "urban"),
        ("sour", "sweet"), ("straight", "crooked"), ("tame", "wild"),
        ("thick", "thin"), ("ugly", "beautiful"), ("vague", "clear"),
        ("warm", "cool"), ("wise", "foolish"), ("guilty", "innocent"),
        ("humble", "proud"), ("lazy", "active"), ("maximum", "minimum"),
        ("narrow", "broad"), ("odd", "even"), ("plain", "fancy"),
        ("rigid", "flexible"), ("solid", "liquid"), ("vacant", "occupied"),
    ]

    # (singular, plural) — irregular forms only
    _PLURALS = [
        ("child", "children"), ("man", "men"), ("woman", "women"),
        ("mouse", "mice"), ("goose", "geese"), ("tooth", "teeth"),
        ("foot", "feet"), ("person", "people"), ("ox", "oxen"),
        ("cactus", "cacti"), ("focus", "foci"), ("fungus", "fungi"),
        ("nucleus", "nuclei"), ("syllabus", "syllabi"), ("analysis", "analyses"),
        ("basis", "bases"), ("crisis", "crises"), ("diagnosis", "diagnoses"),
        ("hypothesis", "hypotheses"), ("thesis", "theses"),
        ("phenomenon", "phenomena"), ("criterion", "criteria"),
        ("datum", "data"), ("medium", "media"), ("bacterium", "bacteria"),
        ("curriculum", "curricula"), ("memorandum", "memoranda"),
        ("appendix", "appendices"), ("index", "indices"), ("matrix", "matrices"),
        ("vertex", "vertices"), ("leaf", "leaves"), ("knife", "knives"),
        ("wife", "wives"), ("life", "lives"), ("wolf", "wolves"),
        ("shelf", "shelves"), ("half", "halves"), ("calf", "calves"),
        ("loaf", "loaves"), ("thief", "thieves"), ("self", "selves"),
        ("deer", "deer"), ("sheep", "sheep"), ("fish", "fish"),
        ("species", "species"), ("series", "series"),
    ]

    # (present, past) — irregular forms only
    _PAST_TENSE = [
        ("go", "went"), ("eat", "ate"), ("run", "ran"), ("see", "saw"),
        ("come", "came"), ("give", "gave"), ("take", "took"), ("make", "made"),
        ("know", "knew"), ("think", "thought"), ("find", "found"),
        ("tell", "told"), ("feel", "felt"), ("become", "became"),
        ("leave", "left"), ("put", "put"), ("mean", "meant"),
        ("keep", "kept"), ("let", "let"), ("begin", "began"),
        ("show", "showed"), ("hear", "heard"), ("play", "played"),
        ("stand", "stood"), ("lose", "lost"), ("pay", "paid"),
        ("meet", "met"), ("bring", "brought"), ("hold", "held"),
        ("write", "wrote"), ("sit", "sat"), ("speak", "spoke"),
        ("read", "read"), ("grow", "grew"), ("spend", "spent"),
        ("win", "won"), ("teach", "taught"), ("buy", "bought"),
        ("send", "sent"), ("fall", "fell"), ("fight", "fought"),
        ("catch", "caught"), ("build", "built"), ("sell", "sold"),
        ("choose", "chose"), ("sleep", "slept"), ("fly", "flew"),
        ("break", "broke"), ("hang", "hung"), ("drive", "drove"),
        ("draw", "drew"), ("sing", "sang"), ("swim", "swam"),
        ("throw", "threw"), ("ride", "rode"), ("ring", "rang"),
        ("drink", "drank"), ("forget", "forgot"), ("blow", "blew"),
        ("wear", "wore"), ("rise", "rose"), ("shake", "shook"),
        ("wake", "woke"), ("freeze", "froze"), ("steal", "stole"),
        ("tear", "tore"), ("dig", "dug"), ("hide", "hid"),
        ("bite", "bit"), ("feed", "fed"), ("lead", "led"),
        ("strike", "struck"), ("sweep", "swept"), ("swing", "swung"),
    ]

    _OPPOSITE_TEMPLATES = [
        "The opposite of {word} is",
        "{word} is the opposite of",
    ]

    _PLURAL_TEMPLATES = [
        "The plural of {word} is",
        "One {word}, many",
    ]

    _PAST_TEMPLATES = [
        "The past tense of {word} is",
        "Today I {word}, yesterday I",
    ]

    def __init__(self, n_prompts: int = 500):
        self.n_prompts = n_prompts

    def load(self) -> list[ProbePrompt]:
        rng = random.Random(42)
        prompts = []

        # Opposites — both directions
        for w1, w2 in self._OPPOSITES:
            for template in self._OPPOSITE_TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word=w1),
                    candidates=[w2],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"transform": "opposite", "input": w1, "output": w2},
                ))
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word=w2),
                    candidates=[w1],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"transform": "opposite", "input": w2, "output": w1},
                ))

        # Plurals
        for singular, plural in self._PLURALS:
            for template in self._PLURAL_TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word=singular),
                    candidates=[plural],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"transform": "plural", "input": singular, "output": plural},
                ))

        # Past tense
        for present, past in self._PAST_TENSE:
            for template in self._PAST_TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word=present),
                    candidates=[past],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"transform": "past_tense", "input": present, "output": past},
                ))

        rng.shuffle(prompts)
        return prompts[:self.n_prompts]
