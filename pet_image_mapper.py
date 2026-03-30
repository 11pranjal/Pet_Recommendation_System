"""
Pet Image Mapper — Maps pet attributes (breed, color, size, age) to local images.

Images live under  petimage/petimage/<breed>/...
The Flask route serves from PETIMAGE_DIR which points to that inner folder.

Mapping rules (per the user's specification):
  ─ German Shepherd  → german/
  ─ Labrador         → labrador/
  ─ Retriever        → golden/
  ─ Spitz            → spitz/
  ─ Pug              → pug/
  ─ Persian           → persian/
  ─ Domestic Shorthair → domestic shorthair/
  ─ Rabbit           → rabbit/
  ─ Parakeet         → parakeet/

Each resolver picks a folder based on the exact combination of
color, AgeMonths (0 vs non-zero), and Size (Small / Medium / Large).
A random image from the chosen folder is assigned deterministically
using  pet_id % len(images).
"""

import os
from urllib.parse import quote

# ── Absolute base path to the inner petimage content folder ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PETIMAGE_DIR = os.path.join(BASE_DIR, "petimage", "petimage")


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _images_in(folder):
    """Return sorted list of image paths (relative to PETIMAGE_DIR),
    scanning the folder recursively to handle nested sub-folders."""
    full = os.path.join(PETIMAGE_DIR, folder)
    if not os.path.isdir(full):
        return []
    imgs = []
    for root, _dirs, files in os.walk(full):
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")) and ":Zone" not in f:
                rel = os.path.relpath(os.path.join(root, f), PETIMAGE_DIR)
                imgs.append(rel.replace("\\", "/"))
    return sorted(imgs)


def _pick(images, pet_id):
    """Deterministically pick one image based on pet_id."""
    if not images:
        return ""
    return images[pet_id % len(images)]


# ═══════════════════════════════════════════════════════════
# GERMAN SHEPHERD  →  german/
# ═══════════════════════════════════════════════════════════
# AgeMonths == 0:
#   color sable           → german/sable-0
#   color starts w/ black → german/german-black0
# AgeMonths != 0:
#   black and red  + Small       → german/blackred small
#   black and red  + Medium/Large → german/blackred adult
#   black and sable + Small      → german/blacksable-small
#   black and sable + Med/Large  → german/blacksable adult
#   black and tan  + Small       → german/blacktan-medium
#   black and tan  + Med/Large   → german/blacktan-old
#   sable          + Small       → german/sable-small
#   sable          + Med/Large   → german/sable-adult

def _resolve_german_shepherd(color, size, age_months):
    c = color.lower().strip()

    if age_months == 0:
        if c == "sable":
            return _images_in("german/sable-0")
        else:  # black and red, black and tan, etc.
            return _images_in("german/german-black0")

    # age != 0
    if "red" in c:  # "black and red"
        if size == "Small":
            return _images_in("german/blackred small")
        else:
            return _images_in("german/blackred adult")

    if c == "black and sable":
        if size == "Small":
            return _images_in("german/blacksable-small")
        else:
            return _images_in("german/blacksable adult")

    if "tan" in c:  # "black and tan"
        if size == "Small":
            return _images_in("german/blacktan-medium")
        else:
            return _images_in("german/blacktan-old")

    if "sable" in c:  # pure "sable"
        if size == "Small":
            return _images_in("german/sable-small")
        else:
            return _images_in("german/sable-adult")

    # fallback
    return _images_in("german/sable-adult")


# ═══════════════════════════════════════════════════════════
# LABRADOR  →  labrador/
# ═══════════════════════════════════════════════════════════
# AgeMonths == 0:
#   black       → labrador/labblack-0
#   grey/gray   → labrador/labgrey-0
#   brown       → labrador/labradorbrown-0
#   white/beige → labrador/labradorwhite-0
# AgeMonths != 0:
#   grey  + Small  → labrador/gray-small
#   grey  + Medium → labrador/gray-adult
#   grey  + Large  → labrador/gray-old
#   black + Small  → labrador/black-small
#   black + Medium → labrador/black-large
#   black + Large  → labrador/black-old
#   brown + Small  → labrador/brown-small
#   brown + Med/Lg → labrador/brown-large
#   white/beige + Small  → labrador/white-small
#   white/beige + Medium → labrador/white-large
#   white/beige + Large  → labrador/white-old

def _resolve_labrador(color, size, age_months):
    c = color.lower().strip()

    if age_months == 0:
        if c == "black":
            return _images_in("labrador/labblack-0")
        elif c in ("grey", "gray"):
            return _images_in("labrador/labgrey-0")
        elif c == "brown":
            return _images_in("labrador/labradorbrown-0")
        elif c in ("white", "beige"):
            return _images_in("labrador/labradorwhite-0")
        else:
            return _images_in("labrador/labblack-0")  # fallback

    # age != 0
    if c in ("grey", "gray"):
        if size == "Small":
            return _images_in("labrador/gray-small")
        elif size == "Medium":
            return _images_in("labrador/gray-adult")
        else:  # Large
            return _images_in("labrador/gray-old")

    if c == "black":
        if size == "Small":
            return _images_in("labrador/black-small")
        elif size == "Medium":
            return _images_in("labrador/black-large")
        else:  # Large
            return _images_in("labrador/black-old")

    if c == "brown":
        if size == "Small":
            return _images_in("labrador/brown-small")
        else:  # Medium or Large
            return _images_in("labrador/brown-large")

    if c in ("white", "beige"):
        if size == "Small":
            return _images_in("labrador/white-small")
        elif size == "Medium":
            return _images_in("labrador/white-large")
        else:  # Large
            return _images_in("labrador/white-old")

    # fallback
    return _images_in("labrador/black-small")


# ═══════════════════════════════════════════════════════════
# RETRIEVER  →  golden/
# ═══════════════════════════════════════════════════════════
# AgeMonths == 0  → golden/golden 0
# AgeMonths != 0:
#   Small  → golden/golden young
#   Medium → golden/golden adult
#   Large  → golden/golden old

def _resolve_retriever(color, size, age_months):
    if age_months == 0:
        return _images_in("golden/golden 0")

    if size == "Small":
        return _images_in("golden/golden young")
    elif size == "Medium":
        return _images_in("golden/golden adult")
    else:  # Large
        return _images_in("golden/golden old")


# ═══════════════════════════════════════════════════════════
# SPITZ  →  spitz/
# ═══════════════════════════════════════════════════════════
# AgeMonths == 0  → spitz/spitz 0
# AgeMonths != 0:
#   Small  → spitz/spitz young
#   Medium → spitz/spitz adult
#   Large  → spitz/spitz old

def _resolve_spitz(color, size, age_months):
    if age_months == 0:
        return _images_in("spitz/spitz 0")

    if size == "Small":
        return _images_in("spitz/spitz young")
    elif size == "Medium":
        return _images_in("spitz/spitz adult")
    else:  # Large
        return _images_in("spitz/spitz old")


# ═══════════════════════════════════════════════════════════
# PUG  →  pug/
# ═══════════════════════════════════════════════════════════
# color black → pug/black/
#   AgeMonths == 0  → bpug 0
#   AgeMonths != 0: Small → bpug young, Medium → bpug adult, Large → bpug old
# color fawn → pug/fawn/
#   AgeMonths == 0  → pug 0
#   AgeMonths != 0: Small → pug young, Medium → pug adult, Large → pug old

def _resolve_pug(color, size, age_months):
    c = color.lower().strip()

    if c == "black":
        if age_months == 0:
            return _images_in("pug/black/bpug 0")
        if size == "Small":
            return _images_in("pug/black/bpug young")
        elif size == "Medium":
            return _images_in("pug/black/bpug adult")
        else:  # Large
            return _images_in("pug/black/bpug old")
    else:  # fawn (default)
        if age_months == 0:
            return _images_in("pug/fawn/pug 0")
        if size == "Small":
            return _images_in("pug/fawn/pug young")
        elif size == "Medium":
            return _images_in("pug/fawn/pug adult")
        else:  # Large
            return _images_in("pug/fawn/pug old")


# ═══════════════════════════════════════════════════════════
# PERSIAN  →  persian/
# ═══════════════════════════════════════════════════════════
# AgeMonths == 0:
#   black  → persian/black 0
#   gray   → persian/gray 0
#   orange → persian/orange 0
#   white  → persian/white 0
# AgeMonths != 0:
#   black  + Small       → persian/black-small
#   black  + Med/Large   → persian/black-large
#   gray   + Small       → persian/gray-small
#   gray   + Med/Large   → persian/gray-large
#   orange + Small       → persian/orange-small
#   orange + Medium      → persian/orange medium
#   orange + Large       → persian/ornage-large   (note: folder typo)
#   white  + Small       → persian/white-small
#   white  + Med/Large   → persian/white-large
# Brown (dataset) → mapped to orange folders as closest match

def _resolve_persian(color, size, age_months):
    c = color.lower().strip()

    # Map dataset "brown" to "orange" folders (no brown folders exist)
    if c == "brown":
        c = "orange"

    if age_months == 0:
        if c == "black":
            return _images_in("persian/black 0")
        elif c in ("gray", "grey"):
            return _images_in("persian/gray 0")
        elif c == "orange":
            return _images_in("persian/orange 0")
        elif c == "white":
            return _images_in("persian/white 0")
        else:
            return _images_in("persian/white 0")  # fallback

    # age != 0
    if c == "black":
        if size == "Small":
            return _images_in("persian/black-small")
        else:  # Medium or Large
            return _images_in("persian/black-large")

    if c in ("gray", "grey"):
        if size == "Small":
            return _images_in("persian/gray-small")
        else:  # Medium or Large
            return _images_in("persian/gray-large")

    if c == "orange":
        if size == "Small":
            return _images_in("persian/orange-small")
        elif size == "Medium":
            return _images_in("persian/orange medium")
        else:  # Large
            return _images_in("persian/ornage-large")  # folder has typo

    if c == "white":
        if size == "Small":
            return _images_in("persian/white-small")
        else:  # Medium or Large
            return _images_in("persian/white-large")

    # fallback
    return _images_in("persian/white-small")


# ═══════════════════════════════════════════════════════════
# DOMESTIC SHORTHAIR  →  domestic shorthair/
# ═══════════════════════════════════════════════════════════
# color white → domestic shorthair/white/
#   AgeMonths == 0  → white 0
#   AgeMonths != 0: Small → white young, Med/Large → white adult old
# color black → domestic shorthair/black/
#   AgeMonths == 0  → black 0
#   AgeMonths != 0: Small → black young, Med/Large → black adult old
# color grey/gray → domestic shorthair/gray/
#   AgeMonths == 0  → gray 0
#   AgeMonths != 0: Small → gray young, Med/Large → garu adult old (folder typo)

def _resolve_domestic_shorthair(color, size, age_months):
    c = color.lower().strip()

    if c == "white":
        if age_months == 0:
            return _images_in("domestic shorthair/white/white 0")
        if size == "Small":
            return _images_in("domestic shorthair/white/white young")
        else:  # Medium or Large
            return _images_in("domestic shorthair/white/white adult old")

    if c == "black":
        if age_months == 0:
            return _images_in("domestic shorthair/black/black 0")
        if size == "Small":
            return _images_in("domestic shorthair/black/black young")
        else:  # Medium or Large
            return _images_in("domestic shorthair/black/black adult old")

    if c in ("grey", "gray"):
        if age_months == 0:
            return _images_in("domestic shorthair/gray/gray 0")
        if size == "Small":
            return _images_in("domestic shorthair/gray/gray young")
        else:  # Medium or Large
            return _images_in("domestic shorthair/gray/garu adult old")  # folder typo

    # fallback
    return _images_in("domestic shorthair/white/white young")


# ═══════════════════════════════════════════════════════════
# RABBIT  →  rabbit/
# ═══════════════════════════════════════════════════════════
# color black → rabbit/black/
#   AgeMonths == 0  → black 0
#   AgeMonths != 0: Small → black young, Med/Large → black adult old
# color white → rabbit/white/
#   AgeMonths == 0  → white 0
#   AgeMonths != 0: Small → white young, Med/Large → white adult old

def _resolve_rabbit(color, size, age_months):
    c = color.lower().strip()

    if c == "black":
        if age_months == 0:
            return _images_in("rabbit/black/black 0")
        if size == "Small":
            return _images_in("rabbit/black/black young")
        else:  # Medium or Large
            return _images_in("rabbit/black/black adult old")

    # white (default)
    if age_months == 0:
        return _images_in("rabbit/white/white 0")
    if size == "Small":
        return _images_in("rabbit/white/white young")
    else:  # Medium or Large
        return _images_in("rabbit/white/white adult old")


# ═══════════════════════════════════════════════════════════
# PARAKEET  →  parakeet/
# ═══════════════════════════════════════════════════════════
# AgeMonths == 0  → parakeet/parakeet 0
# AgeMonths != 0  → parakeet/parakeet young adult old

def _resolve_parakeet(color, size, age_months):
    if age_months == 0:
        return _images_in("parakeet/parakeet 0")
    else:
        return _images_in("parakeet/parakeet young adult old")


# ═══════════════════════════════════════════════════════════
# Breed → resolver dispatch
# ═══════════════════════════════════════════════════════════

_BREED_RESOLVERS = {
    "German Shepherd":    _resolve_german_shepherd,
    "Labrador":           _resolve_labrador,
    "Retriever":          _resolve_retriever,
    "Spitz":              _resolve_spitz,
    "Pug":                _resolve_pug,
    "Persian":            _resolve_persian,
    "Domestic Shorthair": _resolve_domestic_shorthair,
    "Rabbit":             _resolve_rabbit,
    "Parakeet":           _resolve_parakeet,
}


# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════

def get_pet_image_url(breed, color, size, age_months, pet_id):
    """
    Return a URL path for a pet image.

    Parameters
    ----------
    breed      : str   e.g. "Labrador", "Persian"
    color      : str   e.g. "Black", "Sable", "Fawn"
    size       : str   e.g. "Small", "Medium", "Large"
    age_months : int   actual age in months
    pet_id     : int   used for deterministic image selection

    Returns
    -------
    str – URL like "/petimage/labrador/black-large/blablarge.png"
          or empty string "" if no image found.
    """
    breed = (breed or "").strip()
    color = (color or "").strip()
    size = (size or "Medium").strip()
    age_months = int(age_months) if age_months is not None else 30
    pet_id = int(pet_id) if pet_id else 0

    resolver = _BREED_RESOLVERS.get(breed)
    if resolver is None:
        return ""

    images = resolver(color, size, age_months)
    if not images:
        return ""

    chosen = _pick(images, pet_id)
    # URL-encode spaces and special chars so browsers can load the path
    return "/petimage/" + quote(chosen)


def get_all_images_for_pet(breed, color, size, age_months):
    """
    Return ALL matching image URLs for a pet's attributes.
    Useful for a gallery or carousel view.
    """
    breed = (breed or "").strip()
    color = (color or "").strip()
    size = (size or "Medium").strip()
    age_months = int(age_months) if age_months is not None else 30

    resolver = _BREED_RESOLVERS.get(breed)
    if resolver is None:
        return []

    images = resolver(color, size, age_months)
    return ["/petimage/" + quote(img) for img in images]


# ═══════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        # German Shepherd
        ("German Shepherd", "Sable", "Small", 0, 14),
        ("German Shepherd", "Sable", "Small", 6, 5),
        ("German Shepherd", "Sable", "Large", 30, 6),
        ("German Shepherd", "Black And Red", "Small", 0, 7),
        ("German Shepherd", "Black And Red", "Small", 6, 8),
        ("German Shepherd", "Black And Red", "Large", 36, 9),
        ("German Shepherd", "Black And Tan", "Small", 0, 10),
        ("German Shepherd", "Black And Tan", "Small", 6, 11),
        ("German Shepherd", "Black And Tan", "Medium", 30, 12),
        # Labrador
        ("Labrador", "Black", "Small", 0, 20),
        ("Labrador", "Black", "Small", 6, 21),
        ("Labrador", "Black", "Medium", 24, 22),
        ("Labrador", "Black", "Large", 48, 23),
        ("Labrador", "Brown", "Small", 6, 24),
        ("Labrador", "Brown", "Large", 48, 25),
        ("Labrador", "Beige", "Small", 6, 26),
        ("Labrador", "Beige", "Medium", 24, 27),
        ("Labrador", "Beige", "Large", 48, 28),
        # Retriever
        ("Retriever", "Orange", "Small", 0, 30),
        ("Retriever", "Beige", "Small", 6, 31),
        ("Retriever", "White", "Medium", 24, 32),
        ("Retriever", "Orange", "Large", 48, 33),
        # Spitz
        ("Spitz", "White", "Small", 0, 40),
        ("Spitz", "White", "Small", 6, 41),
        ("Spitz", "White", "Medium", 30, 42),
        ("Spitz", "White", "Large", 90, 43),
        # Pug
        ("Pug", "Black", "Small", 0, 50),
        ("Pug", "Black", "Small", 6, 51),
        ("Pug", "Black", "Medium", 30, 52),
        ("Pug", "Fawn", "Small", 0, 53),
        ("Pug", "Fawn", "Large", 48, 54),
        # Persian
        ("Persian", "Black", "Small", 0, 60),
        ("Persian", "Black", "Small", 6, 61),
        ("Persian", "Black", "Large", 48, 62),
        ("Persian", "Orange", "Small", 6, 63),
        ("Persian", "Orange", "Medium", 24, 64),
        ("Persian", "Orange", "Large", 48, 65),
        ("Persian", "White", "Small", 6, 66),
        ("Persian", "Gray", "Large", 48, 67),
        ("Persian", "Brown", "Medium", 24, 68),
        # Domestic Shorthair
        ("Domestic Shorthair", "Black", "Small", 0, 70),
        ("Domestic Shorthair", "Black", "Small", 6, 71),
        ("Domestic Shorthair", "Grey", "Large", 80, 72),
        ("Domestic Shorthair", "White", "Medium", 24, 73),
        # Rabbit
        ("Rabbit", "Black", "Medium", 0, 80),
        ("Rabbit", "Black", "Medium", 24, 81),
        ("Rabbit", "White", "Large", 48, 82),
        # Parakeet
        ("Parakeet", "Green", "Small", 0, 90),
        ("Parakeet", "Green", "Medium", 12, 91),
    ]

    print("=" * 80)
    print("PET IMAGE MAPPER — TEST")
    print("=" * 80)
    for breed, color, size, age, pid in test_cases:
        url = get_pet_image_url(breed, color, size, age, pid)
        status = "✅" if url else "❌"
        print(f"{status} {breed:25s} | {color:15s} | {size:6s} | age={age:3d}m | → {url or '(none)'}")
    print("=" * 80)
