"""
Client-written class descriptions for the RT protocol (simulation of what each owner would type).

Each client writes KEYWORDS (default, keywords()) or a sentence (describe()) for each of ITS OWN classes, in ITS OWN language, knowing nothing
about the other clients. Nothing here is shared: the only thing agreed in advance is the public
anchor list (rt_protocol.TEXT_ANCHORS).

describe(dataset, class_name, lang) -> str
Languages: en, zh, es, ja, fr, de. Unknown datasets / names fall back to "<name>" in a language template.
Override with a JSON file (yaml key rt_descriptions): {"<client_id>": {"<local_id>": "text", ...}, ...}.
"""
DIGIT = {
    "en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    "zh": ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"],
    "es": ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve"],
    "ja": ["ゼロ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう"],
    "fr": ["zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf"],
    "de": ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"],
}
DIGIT_T = {"en": "the handwritten digit {}", "zh": "手寫數字{}", "es": "el dígito escrito a mano {}",
           "ja": "手書きの数字の{}", "fr": "le chiffre manuscrit {}", "de": "die handgeschriebene Ziffer {}"}
UPPER_T = {"en": "the handwritten capital letter {}", "zh": "手寫大寫英文字母{}",
           "es": "la letra mayúscula escrita a mano {}", "ja": "手書きの大文字の{}",
           "fr": "la lettre majuscule manuscrite {}", "de": "der handgeschriebene Großbuchstabe {}"}
LOWER_T = {"en": "the handwritten small letter {}", "zh": "手寫小寫英文字母{}",
           "es": "la letra minúscula escrita a mano {}", "ja": "手書きの小文字の{}",
           "fr": "la lettre minuscule manuscrite {}", "de": "der handgeschriebene Kleinbuchstabe {}"}
PHOTO_T = {"en": "a photo of {}", "zh": "一張{}的照片", "es": "una foto de {}", "ja": "{}の写真",
           "fr": "une photo {}", "de": "ein Foto von {}"}
CIFAR10 = {
    "airplane":   {"zh": "飛機", "es": "un avión", "ja": "飛行機", "fr": "d'un avion", "de": "einem Flugzeug"},
    "automobile": {"zh": "汽車", "es": "un coche", "ja": "自動車", "fr": "d'une voiture", "de": "einem Auto"},
    "bird":       {"zh": "鳥", "es": "un pájaro", "ja": "鳥", "fr": "d'un oiseau", "de": "einem Vogel"},
    "cat":        {"zh": "貓", "es": "un gato", "ja": "猫", "fr": "d'un chat", "de": "einer Katze"},
    "deer":       {"zh": "鹿", "es": "un ciervo", "ja": "鹿", "fr": "d'un cerf", "de": "einem Hirsch"},
    "dog":        {"zh": "狗", "es": "un perro", "ja": "犬", "fr": "d'un chien", "de": "einem Hund"},
    "frog":       {"zh": "青蛙", "es": "una rana", "ja": "カエル", "fr": "d'une grenouille", "de": "einem Frosch"},
    "horse":      {"zh": "馬", "es": "un caballo", "ja": "馬", "fr": "d'un cheval", "de": "einem Pferd"},
    "ship":       {"zh": "船", "es": "un barco", "ja": "船", "fr": "d'un bateau", "de": "einem Schiff"},
    "truck":      {"zh": "卡車", "es": "un camión", "ja": "トラック", "fr": "d'un camion", "de": "einem Lastwagen"},
}
LANGS = list(DIGIT)

# ---- keywords: [domain, (case,) identity], each encoded separately (never joined into a sentence)
KW_DIGIT = {"en": "digit", "zh": "數字", "es": "dígito", "ja": "数字", "fr": "chiffre", "de": "Ziffer"}
KW_LETTER = {"en": "letter", "zh": "字母", "es": "letra", "ja": "文字", "fr": "lettre", "de": "Buchstabe"}
KW_UPPER = {"en": "capital", "zh": "大寫", "es": "mayúscula", "ja": "大文字", "fr": "majuscule", "de": "Großbuchstabe"}
KW_LOWER = {"en": "lowercase", "zh": "小寫", "es": "minúscula", "ja": "小文字", "fr": "minuscule", "de": "Kleinbuchstabe"}
KW_PHOTO = {"en": "photo", "zh": "照片", "es": "foto", "ja": "写真", "fr": "photo", "de": "Foto"}
CIFAR10_NOUN = {
    "airplane":   {"zh": "飛機", "es": "avión", "ja": "飛行機", "fr": "avion", "de": "Flugzeug"},
    "automobile": {"zh": "汽車", "es": "coche", "ja": "自動車", "fr": "voiture", "de": "Auto"},
    "bird":       {"zh": "鳥", "es": "pájaro", "ja": "鳥", "fr": "oiseau", "de": "Vogel"},
    "cat":        {"zh": "貓", "es": "gato", "ja": "猫", "fr": "chat", "de": "Katze"},
    "deer":       {"zh": "鹿", "es": "ciervo", "ja": "鹿", "fr": "cerf", "de": "Hirsch"},
    "dog":        {"zh": "狗", "es": "perro", "ja": "犬", "fr": "chien", "de": "Hund"},
    "frog":       {"zh": "青蛙", "es": "rana", "ja": "カエル", "fr": "grenouille", "de": "Frosch"},
    "horse":      {"zh": "馬", "es": "caballo", "ja": "馬", "fr": "cheval", "de": "Pferd"},
    "ship":       {"zh": "船", "es": "barco", "ja": "船", "fr": "bateau", "de": "Schiff"},
    "truck":      {"zh": "卡車", "es": "camión", "ja": "トラック", "fr": "camion", "de": "Lastwagen"},
}


def keywords(dataset, name, lang):
    """Keyword list for one class, in the client's own language; each item is encoded independently."""
    name = str(name)
    if name.isdigit() and len(name) == 1:
        return [KW_DIGIT[lang], DIGIT[lang][int(name)]]
    if len(name) == 1 and name.isalpha():
        return [KW_LETTER[lang], (KW_UPPER if name.isupper() else KW_LOWER)[lang], name]
    if name in CIFAR10_NOUN:
        return [KW_PHOTO[lang], name if lang == "en" else CIFAR10_NOUN[name][lang]]
    return [KW_PHOTO[lang], name]


def describe(dataset, name, lang):
    name = str(name)
    if name.isdigit() and len(name) == 1:                         # MNIST / USPS / EMNIST digits
        return DIGIT_T[lang].format(DIGIT[lang][int(name)])
    if len(name) == 1 and name.isalpha():                         # EMNIST letters
        return (UPPER_T if name.isupper() else LOWER_T)[lang].format(name)
    if name in CIFAR10:
        return PHOTO_T[lang].format(("an " if name[0] in "aeiou" else "a ") + name if lang == "en" else CIFAR10[name][lang])
    return PHOTO_T[lang].format(name)                             # unknown: name inside a template


if __name__ == "__main__":
    for lang in LANGS:
        print(lang, "|", keywords("MNIST", "3", lang), "|", keywords("EMNIST", "g", lang), "|",
              keywords("CIFAR10", "truck", lang))
