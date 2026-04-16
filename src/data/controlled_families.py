"""Controlled concept families for cross-concept steerability analysis.

All families use the SAME task structure to eliminate structural confounds:
- Binary classification: each prompt belongs to class A or class B
- Balanced: ~50/50 split between A and B
- Same prompt format: "{entity} {relation}"
- Steering target: always class A
- Same n_target and n_other per family

This isolates representation geometry as the only variable across families.
"""

import random
from src.data.base import ProbeDataset, ProbePrompt


class ControlledProbeFamily(ProbeDataset):
    """Base class for controlled binary concept families."""

    family_name = ""
    _CLASS_A_LABEL = ""
    _CLASS_B_LABEL = ""
    _DATA_A = []  # list of (prompt_text, ) for class A
    _DATA_B = []  # list of (prompt_text, ) for class B
    _TEMPLATES = ["{}"]

    def __init__(self, n_prompts=500):
        self.n_prompts = n_prompts

    def load(self):
        prompts = []
        for prompt_text in self._DATA_A:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(prompt_text),
                    candidates=[self._CLASS_A_LABEL],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"class": "A", "label": self._CLASS_A_LABEL},
                ))
        for prompt_text in self._DATA_B:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(prompt_text),
                    candidates=[self._CLASS_B_LABEL],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"class": "B", "label": self._CLASS_B_LABEL},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# PARITY: even vs odd numbers
# ============================================================

class ParityProbe(ControlledProbeFamily):
    """Is the arithmetic result even or odd?"""
    family_name = "c_parity"
    _CLASS_A_LABEL = "even"
    _CLASS_B_LABEL = "odd"
    _TEMPLATES = [
        "The number {} is",
        "{} is an",
        "Is {} even or odd? It is",
    ]

    def load(self):
        prompts = []
        for n in range(2, 100):
            label = "even" if n % 2 == 0 else "odd"
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(n),
                    candidates=[label],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"number": n, "class": label},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# CONTINENT: Europe vs Asia
# ============================================================

class ContinentBinaryProbe(ControlledProbeFamily):
    """Is this country in Europe or Asia?"""
    family_name = "c_continent"
    _CLASS_A_LABEL = "Europe"
    _CLASS_B_LABEL = "Asia"

    _EUROPE = [
        "France", "Germany", "Spain", "Italy", "Poland", "Sweden",
        "Greece", "Portugal", "Ireland", "Norway", "Finland", "Denmark",
        "Austria", "Belgium", "Netherlands", "Switzerland", "Romania",
        "Hungary", "Czechia", "Croatia", "Serbia", "Bulgaria", "Ukraine",
    ]

    _ASIA = [
        "Japan", "China", "Korea", "Thailand", "Vietnam", "Indonesia",
        "India", "Iran", "Iraq", "Turkey", "Philippines", "Taiwan",
        "Pakistan", "Bangladesh", "Malaysia", "Myanmar", "Nepal",
        "Mongolia", "Cambodia", "Laos", "Singapore", "Afghanistan",
    ]

    _TEMPLATES = [
        "{} is a country in",
        "{} is located in",
        "The continent of {} is",
        "{} belongs to the continent of",
    ]

    def load(self):
        prompts = []
        for country in self._EUROPE:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(country),
                    candidates=["Europe"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"country": country, "class": "Europe"},
                ))
        for country in self._ASIA:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(country),
                    candidates=["Asia"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"country": country, "class": "Asia"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# ANIMAL CLASS: mammal vs bird
# ============================================================

class AnimalBinaryProbe(ControlledProbeFamily):
    """Is this animal a mammal or a bird?"""
    family_name = "c_animal"
    _CLASS_A_LABEL = "mammal"
    _CLASS_B_LABEL = "bird"

    _MAMMALS = [
        "dog", "cat", "horse", "cow", "whale", "dolphin", "bat",
        "elephant", "lion", "bear", "monkey", "rabbit", "deer",
        "wolf", "fox", "tiger", "pig", "sheep", "goat", "mouse",
        "rat", "hamster", "squirrel", "giraffe", "zebra",
    ]

    _BIRDS = [
        "robin", "eagle", "penguin", "parrot", "owl", "sparrow",
        "hawk", "duck", "swan", "crow", "pigeon", "falcon",
        "flamingo", "pelican", "heron", "seagull", "woodpecker",
        "cardinal", "bluebird", "dove", "stork", "crane",
        "ostrich", "toucan", "hummingbird",
    ]

    _TEMPLATES = [
        "A {} is a",
        "The {} is a",
        "In biology, a {} is classified as a",
        "A {} belongs to the group of",
    ]

    def load(self):
        prompts = []
        for animal in self._MAMMALS:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(animal),
                    candidates=["mammal"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"animal": animal, "class": "mammal"},
                ))
        for animal in self._BIRDS:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(animal),
                    candidates=["bird"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"animal": animal, "class": "bird"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# GENDER: he vs she (grammatical, from occupation stereotypes)
# ============================================================

class GenderProbe(ControlledProbeFamily):
    """Grammatical gender: does this name typically use he or she?"""
    family_name = "c_gender"
    _CLASS_A_LABEL = "he"
    _CLASS_B_LABEL = "she"

    _MALE_NAMES = [
        "John", "James", "Robert", "Michael", "William", "David",
        "Richard", "Joseph", "Thomas", "Charles", "Daniel", "Matthew",
        "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew",
        "Joshua", "Kenneth", "Kevin", "Brian", "George", "Timothy",
    ]

    _FEMALE_NAMES = [
        "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth",
        "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy",
        "Betty", "Margaret", "Sandra", "Ashley", "Dorothy", "Kimberly",
        "Emily", "Donna", "Michelle", "Carol", "Amanda", "Melissa",
    ]

    _TEMPLATES = [
        "{name} said that",
        "{name} went to the store and",
        "When {name} arrived,",
        "{name} picked up the book and",
    ]

    def load(self):
        prompts = []
        for name in self._MALE_NAMES:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(name=name),
                    candidates=["he"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"name": name, "class": "male"},
                ))
        for name in self._FEMALE_NAMES:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(name=name),
                    candidates=["she"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"name": name, "class": "female"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# LIVING: alive vs not alive
# ============================================================

class LivingProbe(ControlledProbeFamily):
    """Is this thing alive or not?"""
    family_name = "c_living"
    _CLASS_A_LABEL = "alive"
    _CLASS_B_LABEL = "not"

    _ALIVE = [
        "A dog is", "A tree is", "A fish is", "A bird is",
        "A cat is", "A flower is", "A horse is", "A frog is",
        "Grass is", "A whale is", "A mushroom is", "A bee is",
        "A snake is", "A spider is", "A worm is", "Coral is",
        "A bear is", "A deer is", "An ant is", "A butterfly is",
        "A dolphin is", "A rose is", "An oak is", "A shark is",
    ]

    _NOT_ALIVE = [
        "A rock is", "A car is", "A table is", "A computer is",
        "Water is", "A book is", "A chair is", "A phone is",
        "A clock is", "A lamp is", "A shoe is", "A key is",
        "A coin is", "A pen is", "A cup is", "A window is",
        "A door is", "A bridge is", "A road is", "A wall is",
        "A mirror is", "A bottle is", "A hammer is", "A nail is",
    ]

    _TEMPLATES = [
        "{} {} considered",
        "{} {}",
    ]

    def load(self):
        prompts = []
        # Simpler: direct prompts
        for prompt in self._ALIVE:
            prompts.append(ProbePrompt(
                prompt_text=prompt, candidates=["alive"], correct_index=0,
                family=self.family_name, metadata={"class": "alive"},
            ))
        for prompt in self._NOT_ALIVE:
            prompts.append(ProbePrompt(
                prompt_text=prompt + " not", candidates=["alive"], correct_index=0,
                family=self.family_name, metadata={"class": "not_alive"},
            ))
        # Actually this is tricky - "A rock is not alive" vs "A dog is alive"
        # Let's use a cleaner format
        prompts = []
        living_things = [
            "dog", "cat", "tree", "fish", "bird", "flower", "horse", "frog",
            "whale", "bee", "snake", "spider", "bear", "deer", "ant",
            "butterfly", "dolphin", "rose", "shark", "turtle", "rabbit",
            "eagle", "wolf", "crab",
        ]
        nonliving_things = [
            "rock", "car", "table", "computer", "book", "chair", "phone",
            "clock", "lamp", "shoe", "key", "coin", "pen", "cup", "window",
            "door", "bridge", "road", "wall", "mirror", "bottle", "hammer",
            "nail", "brick",
        ]
        for thing in living_things:
            for template in [
                "A {} is a living",
                "The {} is a living",
                "A {} is considered a living",
            ]:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["thing"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "living"},
                ))
        for thing in nonliving_things:
            for template in [
                "A {} is a non-living",
                "The {} is a non-living",
                "A {} is considered a non-living",
            ]:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["thing"],
                    correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "nonliving"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# TEMPERATURE: hot vs cold
# ============================================================

class TemperatureProbe(ControlledProbeFamily):
    """Is this thing hot or cold?"""
    family_name = "c_temperature"
    _CLASS_A_LABEL = "hot"
    _CLASS_B_LABEL = "cold"

    _HOT = [
        "fire", "lava", "sun", "oven", "boiling water", "desert",
        "summer", "flame", "furnace", "sauna", "coffee", "tea",
        "steam", "iron", "stove", "torch", "ember", "volcano",
        "fireplace", "heater", "chili pepper", "hot springs",
    ]

    _COLD = [
        "ice", "snow", "glacier", "winter", "freezer", "Antarctica",
        "Arctic", "frost", "blizzard", "iceberg", "frozen lake",
        "ice cream", "snowflake", "refrigerator", "cold wind",
        "hail", "permafrost", "icicle", "north pole", "avalanche",
        "sleet", "igloo",
    ]

    _TEMPLATES = [
        "{} is",
        "The word {} is associated with something",
        "{} is typically considered",
        "You would describe {} as",
    ]

    def load(self):
        prompts = []
        for thing in self._HOT:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["hot"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "hot"},
                ))
        for thing in self._COLD:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["cold"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "cold"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# SIZE: big vs small
# ============================================================

class SizeProbe(ControlledProbeFamily):
    """Is this thing big or small?"""
    family_name = "c_size"
    _CLASS_A_LABEL = "big"
    _CLASS_B_LABEL = "small"

    _BIG = [
        "elephant", "whale", "mountain", "ocean", "skyscraper", "planet",
        "dinosaur", "continent", "galaxy", "stadium", "cathedral",
        "aircraft carrier", "redwood tree", "giraffe", "hippopotamus",
        "blue whale", "great wall", "pyramid", "volcano", "glacier",
        "castle", "palace",
    ]

    _SMALL = [
        "ant", "grain of sand", "atom", "needle", "coin", "button",
        "seed", "pebble", "raindrop", "snowflake", "ladybug", "flea",
        "grain of rice", "pinhead", "mouse", "hummingbird", "berry",
        "thimble", "dust", "crumb", "mite", "bacteria",
    ]

    _TEMPLATES = [
        "A {} is",
        "Compared to most things, a {} is",
        "In terms of size, a {} is",
        "A {} is considered",
    ]

    def load(self):
        prompts = []
        for thing in self._BIG:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["big"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "big"},
                ))
        for thing in self._SMALL:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["small"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "small"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# SPEED: fast vs slow
# ============================================================

class SpeedProbe(ControlledProbeFamily):
    """Is this thing fast or slow?"""
    family_name = "c_speed"
    _CLASS_A_LABEL = "fast"
    _CLASS_B_LABEL = "slow"

    _FAST = [
        "cheetah", "rocket", "light", "bullet", "jet", "lightning",
        "falcon", "race car", "comet", "laser", "sports car",
        "gazelle", "speedboat", "fighter jet", "missile",
        "hare", "roadrunner", "tornado", "express train",
        "hawk", "eagle", "greyhound",
    ]

    _SLOW = [
        "snail", "turtle", "sloth", "glacier", "tree growth",
        "tortoise", "slug", "caterpillar", "starfish",
        "manatee", "coral growth", "continental drift",
        "stalactite", "erosion", "tectonic plate",
        "garden snail", "sea cucumber", "earthworm",
        "koala", "loris", "three-toed sloth", "sea horse",
    ]

    _TEMPLATES = [
        "A {} is",
        "In terms of speed, a {} is",
        "A {} is considered",
        "Compared to most things, a {} is",
    ]

    def load(self):
        prompts = []
        for thing in self._FAST:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["fast"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "fast"},
                ))
        for thing in self._SLOW:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["slow"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "slow"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# MATERIAL: metal vs wood
# ============================================================

class MaterialProbe(ControlledProbeFamily):
    """Is this made of metal or wood?"""
    family_name = "c_material"
    _CLASS_A_LABEL = "metal"
    _CLASS_B_LABEL = "wood"

    _METAL = [
        "sword", "coin", "nail", "car", "fork", "knife", "spoon",
        "key", "lock", "chain", "anchor", "bell", "shield",
        "helmet", "cannon", "locomotive", "bridge", "skyscraper",
        "submarine", "tank", "razor", "scissors",
    ]

    _WOOD = [
        "table", "chair", "door", "floor", "bookshelf", "cabinet",
        "barrel", "violin", "guitar", "bat", "canoe", "log",
        "fence", "deck", "beam", "plank", "crate", "pew",
        "totem pole", "birdhouse", "treehouse", "raft",
    ]

    _TEMPLATES = [
        "A {} is typically made of",
        "The primary material of a {} is",
        "A {} is usually constructed from",
        "A traditional {} is made of",
    ]

    def load(self):
        prompts = []
        for thing in self._METAL:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["metal"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "metal"},
                ))
        for thing in self._WOOD:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["wood"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "wood"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# WATER: land vs water
# ============================================================

class HabitatProbe(ControlledProbeFamily):
    """Does this animal live on land or in water?"""
    family_name = "c_habitat"
    _CLASS_A_LABEL = "land"
    _CLASS_B_LABEL = "water"

    _LAND = [
        "dog", "cat", "horse", "elephant", "lion", "tiger", "bear",
        "deer", "rabbit", "wolf", "fox", "monkey", "snake", "lizard",
        "cow", "pig", "sheep", "goat", "chicken", "ant", "spider",
        "eagle",
    ]

    _WATER = [
        "fish", "whale", "dolphin", "shark", "octopus", "jellyfish",
        "seahorse", "clam", "lobster", "crab", "shrimp", "starfish",
        "squid", "eel", "ray", "salmon", "tuna", "cod", "trout",
        "swordfish", "barracuda", "catfish",
    ]

    _TEMPLATES = [
        "A {} lives on",
        "The natural habitat of a {} is",
        "A {} is found on",
        "You would find a {} on",
    ]

    def load(self):
        prompts = []
        for animal in self._LAND:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(animal),
                    candidates=["land"], correct_index=0,
                    family=self.family_name,
                    metadata={"animal": animal, "class": "land"},
                ))
        for animal in self._WATER:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(animal),
                    candidates=["water"], correct_index=0,
                    family=self.family_name,
                    metadata={"animal": animal, "class": "water"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# INDOOR vs OUTDOOR
# ============================================================

class IndoorOutdoorProbe(ControlledProbeFamily):
    """Is this activity/thing indoor or outdoor?"""
    family_name = "c_indoor"
    _CLASS_A_LABEL = "indoor"
    _CLASS_B_LABEL = "outdoor"

    _INDOOR = [
        "cooking", "reading", "sleeping", "watching TV", "typing",
        "showering", "ironing", "vacuuming", "dishwashing",
        "studying", "knitting", "painting indoors", "baking",
        "playing chess", "doing laundry", "cleaning",
        "playing piano", "eating dinner", "working at a desk",
        "taking a bath", "folding clothes", "washing dishes",
    ]

    _OUTDOOR = [
        "hiking", "swimming", "camping", "gardening", "skiing",
        "surfing", "fishing", "running", "cycling", "sailing",
        "rock climbing", "birdwatching", "horseback riding",
        "kayaking", "skateboarding", "jogging",
        "playing soccer", "flying a kite", "barbecuing",
        "stargazing", "sunbathing", "snowboarding",
    ]

    _TEMPLATES = [
        "{} is typically an",
        "{} is usually done",
        "{} is an activity done",
        "You would typically do {} while",
    ]

    def load(self):
        prompts = []
        for activity in self._INDOOR:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(activity),
                    candidates=["indoor"], correct_index=0,
                    family=self.family_name,
                    metadata={"activity": activity, "class": "indoor"},
                ))
        for activity in self._OUTDOOR:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(activity),
                    candidates=["outdoor"], correct_index=0,
                    family=self.family_name,
                    metadata={"activity": activity, "class": "outdoor"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# EDIBLE: edible vs not edible
# ============================================================

class EdibleProbe(ControlledProbeFamily):
    """Is this thing edible or not?"""
    family_name = "c_edible"

    _EDIBLE = [
        "apple", "bread", "cheese", "rice", "chicken", "carrot",
        "potato", "banana", "egg", "fish", "tomato", "corn",
        "strawberry", "chocolate", "honey", "milk", "butter",
        "pasta", "pizza", "cake", "cookie", "grape", "mango",
    ]

    _INEDIBLE = [
        "rock", "plastic", "glass", "steel", "rubber", "concrete",
        "paper", "sand", "brick", "wire", "battery", "gasoline",
        "paint", "glue", "chalk", "soap", "wax", "clay",
        "marble", "gravel", "asphalt", "cement", "silicon",
    ]

    _TEMPLATES = [
        "A {} is",
        "{} is generally considered",
        "You would describe {} as",
        "{} is typically",
    ]

    def load(self):
        prompts = []
        for thing in self._EDIBLE:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["edible"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "edible"},
                ))
        for thing in self._INEDIBLE:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["inedible"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "inedible"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# NATURAL vs MAN-MADE
# ============================================================

class NaturalProbe(ControlledProbeFamily):
    """Is this thing natural or man-made?"""
    family_name = "c_natural"

    _NATURAL = [
        "mountain", "river", "tree", "cloud", "ocean", "forest",
        "volcano", "waterfall", "desert", "glacier", "cave", "island",
        "meadow", "canyon", "coral reef", "geyser", "aurora",
        "rainbow", "earthquake", "tornado", "hurricane", "sunrise",
    ]

    _MANMADE = [
        "bridge", "skyscraper", "road", "dam", "tunnel", "factory",
        "airport", "stadium", "lighthouse", "pyramid", "canal",
        "railroad", "telescope", "satellite", "subway", "highway",
        "aqueduct", "windmill", "castle", "monument", "antenna",
        "pipeline",
    ]

    _TEMPLATES = [
        "A {} is",
        "The {} is considered",
        "A {} is classified as",
        "You would describe a {} as",
    ]

    def load(self):
        prompts = []
        for thing in self._NATURAL:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["natural"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "natural"},
                ))
        for thing in self._MANMADE:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["artificial"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "artificial"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# SOLID vs LIQUID
# ============================================================

class PhaseProbe(ControlledProbeFamily):
    """Is this substance typically solid or liquid?"""
    family_name = "c_phase"

    _SOLID = [
        "rock", "ice", "wood", "iron", "glass", "diamond",
        "bone", "brick", "concrete", "gold", "silver", "copper",
        "stone", "marble", "steel", "plastic", "rubber", "salt",
        "sugar", "sand", "chalk", "granite",
    ]

    _LIQUID = [
        "water", "oil", "milk", "juice", "blood", "wine",
        "vinegar", "honey", "gasoline", "mercury", "soup",
        "coffee", "tea", "alcohol", "syrup", "broth",
        "paint", "ink", "perfume", "acid", "seawater", "rain",
    ]

    _TEMPLATES = [
        "At room temperature, {} is a",
        "{} is typically a",
        "The physical state of {} is",
        "{} is normally a",
    ]

    def load(self):
        prompts = []
        for thing in self._SOLID:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["solid"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "solid"},
                ))
        for thing in self._LIQUID:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["liquid"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "liquid"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# PLANT vs ANIMAL
# ============================================================

class PlantAnimalProbe(ControlledProbeFamily):
    """Is this a plant or an animal?"""
    family_name = "c_plant"

    _PLANT = [
        "oak", "rose", "tulip", "daisy", "sunflower", "cactus",
        "fern", "bamboo", "ivy", "moss", "orchid", "lily",
        "pine", "maple", "willow", "palm", "clover", "seaweed",
        "lavender", "mint", "basil", "thyme",
    ]

    _ANIMAL = [
        "dog", "cat", "eagle", "whale", "shark", "ant",
        "elephant", "lion", "snake", "frog", "bear", "wolf",
        "dolphin", "penguin", "butterfly", "spider", "rabbit",
        "horse", "tiger", "crow", "owl", "salmon",
    ]

    _TEMPLATES = [
        "A {} is a",
        "The {} is classified as a",
        "In biology, a {} is a",
        "A {} belongs to the kingdom of",
    ]

    def load(self):
        prompts = []
        for thing in self._PLANT:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["plant"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "plant"},
                ))
        for thing in self._ANIMAL:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["animal"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "animal"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# SINGULAR vs PLURAL
# ============================================================

class PluralProbe(ControlledProbeFamily):
    """Is this word singular or plural?"""
    family_name = "c_plural"

    _SINGULAR = [
        "cat", "dog", "house", "tree", "child", "book",
        "city", "car", "bird", "fish", "mouse", "person",
        "woman", "man", "foot", "tooth", "leaf", "knife",
        "box", "church", "bus", "wish",
    ]

    _PLURAL = [
        "cats", "dogs", "houses", "trees", "children", "books",
        "cities", "cars", "birds", "fishes", "mice", "people",
        "women", "men", "feet", "teeth", "leaves", "knives",
        "boxes", "churches", "buses", "wishes",
    ]

    _TEMPLATES_S = [
        "The word '{}' is",
        "'{}' is a",
        "Grammatically, '{}' is",
        "The noun '{}' is in the",
    ]

    def load(self):
        prompts = []
        for word in self._SINGULAR:
            for template in self._TEMPLATES_S:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word),
                    candidates=["singular"], correct_index=0,
                    family=self.family_name,
                    metadata={"word": word, "class": "singular"},
                ))
        for word in self._PLURAL:
            for template in self._TEMPLATES_S:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word),
                    candidates=["plural"], correct_index=0,
                    family=self.family_name,
                    metadata={"word": word, "class": "plural"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# POSITIVE vs NEGATIVE sentiment
# ============================================================

class SentimentProbe(ControlledProbeFamily):
    """Is this word positive or negative?"""
    family_name = "c_sentiment"

    _POSITIVE = [
        "happy", "beautiful", "wonderful", "excellent", "amazing",
        "brilliant", "fantastic", "great", "lovely", "perfect",
        "delightful", "joyful", "magnificent", "superb", "outstanding",
        "glorious", "marvelous", "splendid", "terrific", "incredible",
        "awesome", "pleasant",
    ]

    _NEGATIVE = [
        "terrible", "horrible", "awful", "dreadful", "disgusting",
        "miserable", "pathetic", "ugly", "cruel", "painful",
        "horrific", "tragic", "devastating", "appalling", "disastrous",
        "wretched", "ghastly", "atrocious", "abysmal", "hideous",
        "vile", "nasty",
    ]

    _TEMPLATES = [
        "The word '{}' has a",
        "'{}' is a",
        "The sentiment of '{}' is",
        "'{}' conveys a",
    ]

    def load(self):
        prompts = []
        for word in self._POSITIVE:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word),
                    candidates=["positive"], correct_index=0,
                    family=self.family_name,
                    metadata={"word": word, "class": "positive"},
                ))
        for word in self._NEGATIVE:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(word),
                    candidates=["negative"], correct_index=0,
                    family=self.family_name,
                    metadata={"word": word, "class": "negative"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# OLD vs NEW / ANCIENT vs MODERN
# ============================================================

class AgeProbe(ControlledProbeFamily):
    """Is this thing ancient/old or modern/new?"""
    family_name = "c_age"

    _OLD = [
        "pyramid", "dinosaur", "fossil", "ruins", "ancient Rome",
        "medieval castle", "stone age", "pharaoh", "chariot",
        "papyrus", "sundial", "catapult", "colosseum", "hieroglyphics",
        "Viking ship", "Roman empire", "feudal lord", "knight",
        "monastery", "quill", "parchment", "abacus",
    ]

    _NEW = [
        "smartphone", "internet", "drone", "electric car", "laptop",
        "social media", "GPS", "robot", "3D printer", "satellite",
        "blockchain", "WiFi", "virtual reality", "AI", "streaming",
        "space station", "microchip", "laser", "MRI",
        "solar panel", "fiber optic", "touchscreen",
    ]

    _TEMPLATES = [
        "A {} is considered",
        "The {} is typically classified as",
        "You would describe a {} as",
        "A {} is",
    ]

    def load(self):
        prompts = []
        for thing in self._OLD:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["ancient"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "ancient"},
                ))
        for thing in self._NEW:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["modern"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "modern"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# DANGEROUS vs SAFE
# ============================================================

class DangerProbe(ControlledProbeFamily):
    """Is this thing dangerous or safe?"""
    family_name = "c_danger"

    _DANGEROUS = [
        "lion", "volcano", "tornado", "shark", "lightning",
        "poison", "earthquake", "wildfire", "avalanche", "cobra",
        "crocodile", "hurricane", "tsunami", "scorpion", "blizzard",
        "landmine", "viper", "piranha", "quicksand", "rattlesnake",
        "grizzly bear", "black widow",
    ]

    _SAFE = [
        "pillow", "blanket", "teddy bear", "kitten", "butterfly",
        "flower", "rainbow", "sunshine", "puppy", "bunny",
        "bubble", "feather", "cotton candy", "marshmallow", "daisy",
        "ladybug", "goldfish", "dove", "lamb", "duckling",
        "hamster", "starlight",
    ]

    _TEMPLATES = [
        "A {} is generally considered",
        "Most people would say a {} is",
        "A {} is typically regarded as",
        "You would classify a {} as",
    ]

    def load(self):
        prompts = []
        for thing in self._DANGEROUS:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["dangerous"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "dangerous"},
                ))
        for thing in self._SAFE:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["safe"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "safe"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# LOUD vs QUIET
# ============================================================

class VolumeProbe(ControlledProbeFamily):
    """Is this thing loud or quiet?"""
    family_name = "c_volume"

    _LOUD = [
        "thunder", "explosion", "siren", "drum", "horn",
        "fireworks", "jackhammer", "concert", "scream", "alarm",
        "jet engine", "motorcycle", "chainsaw", "trumpet",
        "whistle", "cannon", "stadium crowd", "tornado siren",
        "ambulance", "rock band", "waterfall", "helicopter",
    ]

    _QUIET = [
        "whisper", "snowfall", "feather", "shadow", "moonlight",
        "library", "meditation", "sleep", "candle", "butterfly",
        "cloud", "silk", "fog", "dew", "moss",
        "starlight", "breeze", "dawn", "sunset", "velvet",
        "bubble", "raindrop",
    ]

    _TEMPLATES = [
        "A {} is typically",
        "The sound of a {} is",
        "You would describe a {} as",
        "A {} is considered",
    ]

    def load(self):
        prompts = []
        for thing in self._LOUD:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["loud"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "loud"},
                ))
        for thing in self._QUIET:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["quiet"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "quiet"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


# ============================================================
# HEAVY vs LIGHT
# ============================================================

# ============================================================
# MID-RANGE FAMILIES: designed to produce A_lin in the 0.1-0.3 range
# These use more abstract/fuzzy classifications that the model
# partially knows but doesn't consistently decode linearly.
# ============================================================


class RoundFlatProbe(ControlledProbeFamily):
    """Is this object round or flat?"""
    family_name = "c_shape"

    _ROUND = [
        "ball", "globe", "sphere", "wheel", "moon", "sun",
        "orange", "apple", "cherry", "marble", "pearl", "bubble",
        "basketball", "baseball", "bowling ball", "eyeball",
        "snowball", "grapefruit", "melon", "coconut",
        "balloon", "planet",
    ]

    _FLAT = [
        "paper", "pancake", "table", "floor", "screen", "mirror",
        "leaf", "coin", "plate", "pizza", "map", "carpet",
        "board", "window", "door", "page", "photograph", "stamp",
        "blanket", "tortilla", "cracker", "wafer",
    ]

    _TEMPLATES = [
        "A {} is typically",
        "The shape of a {} is",
        "You would describe a {} as",
        "A {} is generally",
    ]

    def load(self):
        prompts = []
        for thing in self._ROUND:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["round"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "round"},
                ))
        for thing in self._FLAT:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["flat"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "flat"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


class HardSoftProbe(ControlledProbeFamily):
    """Is this thing hard or soft?"""
    family_name = "c_hardness"

    _HARD = [
        "diamond", "steel", "rock", "glass", "bone", "concrete",
        "iron", "brick", "marble", "granite", "ice", "shell",
        "crystal", "stone", "metal", "ceramic", "titanium",
        "bronze", "amber", "obsidian", "jade", "quartz",
    ]

    _SOFT = [
        "pillow", "cotton", "silk", "fur", "foam", "butter",
        "cloud", "marshmallow", "velvet", "feather", "wool",
        "sponge", "dough", "clay", "mud", "jelly",
        "cushion", "blanket", "plush", "fleece", "moss", "cashmere",
    ]

    _TEMPLATES = [
        "A {} is",
        "The texture of {} is",
        "You would describe {} as",
        "{} is typically considered",
    ]

    def load(self):
        prompts = []
        for thing in self._HARD:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["hard"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "hard"},
                ))
        for thing in self._SOFT:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["soft"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "soft"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


class WetDryProbe(ControlledProbeFamily):
    """Is this thing wet or dry?"""
    family_name = "c_moisture"

    _WET = [
        "ocean", "rain", "river", "waterfall", "swamp", "puddle",
        "dew", "fog", "mist", "steam", "tears", "sweat",
        "ice cream", "soup", "juice", "blood", "saliva",
        "mud", "slime", "paint", "oil", "sap",
    ]

    _DRY = [
        "desert", "dust", "sand", "bone", "paper", "chalk",
        "flour", "powder", "ash", "straw", "hay", "cork",
        "toast", "cracker", "sawdust", "gravel", "salt",
        "parchment", "cotton ball", "charcoal", "cinnamon",
        "biscuit",
    ]

    _TEMPLATES = [
        "{} is typically",
        "You would describe {} as",
        "{} is generally considered",
        "The typical state of {} is",
    ]

    def load(self):
        prompts = []
        for thing in self._WET:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["wet"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "wet"},
                ))
        for thing in self._DRY:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["dry"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "dry"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


class DayNightProbe(ControlledProbeFamily):
    """Is this associated with day or night?"""
    family_name = "c_daynight"

    _DAY = [
        "sunshine", "sunrise", "morning", "noon", "afternoon",
        "lunch", "breakfast", "daylight", "blue sky", "sunlight",
        "rooster", "office work", "school", "commute", "picnic",
        "beach", "gardening", "shopping", "playground", "park",
        "farmers market", "jogging",
    ]

    _NIGHT = [
        "moonlight", "stars", "midnight", "darkness", "sunset",
        "dinner", "sleep", "dreams", "owl", "bat",
        "nightclub", "fireworks", "campfire", "candle", "lantern",
        "astronomy", "stargazing", "lullaby", "pajamas", "pillow",
        "bedtime", "nightgown",
    ]

    _TEMPLATES = [
        "{} is associated with",
        "{} typically happens during the",
        "You would associate {} with",
        "{} is most common during the",
    ]

    def load(self):
        prompts = []
        for thing in self._DAY:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["day"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "day"},
                ))
        for thing in self._NIGHT:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["night"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "night"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


class SweetSourProbe(ControlledProbeFamily):
    """Is this taste sweet or sour?"""
    family_name = "c_taste"

    _SWEET = [
        "sugar", "honey", "chocolate", "candy", "cake", "cookie",
        "ice cream", "maple syrup", "caramel", "fruit",
        "marshmallow", "jelly", "jam", "brownie", "fudge",
        "pudding", "cupcake", "donut", "waffle", "milkshake",
        "cotton candy", "melon",
    ]

    _SOUR = [
        "lemon", "vinegar", "lime", "grapefruit", "pickle",
        "yogurt", "sauerkraut", "cranberry", "tamarind",
        "green apple", "citrus", "kumquat", "rhubarb",
        "sourdough", "buttermilk", "kefir", "kombucha",
        "gooseberry", "cider vinegar", "kimchi", "unripe fruit",
        "fermented cabbage",
    ]

    _TEMPLATES = [
        "{} tastes",
        "The flavor of {} is",
        "{} is typically",
        "You would describe the taste of {} as",
    ]

    def load(self):
        prompts = []
        for thing in self._SWEET:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["sweet"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "sweet"},
                ))
        for thing in self._SOUR:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["sour"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "sour"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]


class WeightProbe(ControlledProbeFamily):
    """Is this thing heavy or light?"""
    family_name = "c_weight"

    _HEAVY = [
        "elephant", "boulder", "truck", "anchor", "locomotive",
        "whale", "tank", "piano", "safe", "anvil",
        "hippo", "concrete block", "refrigerator", "cannon",
        "ship", "tractor", "submarine", "bulldozer",
        "statue", "iron beam", "oak log", "lead brick",
    ]

    _LIGHT = [
        "feather", "leaf", "butterfly", "snowflake", "bubble",
        "paper", "balloon", "cotton ball", "dandelion seed", "dust",
        "soap bubble", "eyelash", "petal", "confetti", "tissue",
        "spider web", "down feather", "ash", "foam", "ribbon",
        "straw", "thread",
    ]

    _TEMPLATES = [
        "A {} is",
        "In terms of weight, a {} is",
        "A {} is considered",
        "Compared to most things, a {} is",
    ]

    def load(self):
        prompts = []
        for thing in self._HEAVY:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["heavy"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "heavy"},
                ))
        for thing in self._LIGHT:
            for template in self._TEMPLATES:
                prompts.append(ProbePrompt(
                    prompt_text=template.format(thing),
                    candidates=["light"], correct_index=0,
                    family=self.family_name,
                    metadata={"thing": thing, "class": "light"},
                ))
        random.Random(42).shuffle(prompts)
        return prompts[:self.n_prompts]
