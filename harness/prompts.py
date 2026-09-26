"""Polish prompts: one instruction and one worked example per answer type.

Examples are deliberately different from the dev set questions, so the dev score is not inflated.
"""
import json
import re

from . import config

SYSTEM = (
    "Jesteś doświadczonym nauczycielem historii i egzaminatorem maturalnym. "
    "Odpowiadasz rzetelnie, zgodnie z faktami historycznymi. "
    "Zawsze kończysz odpowiedź linią w wymaganym formacie."
)

FORMAT = {
    "single": "W ostatniej linii napisz: Odpowiedź: <jedna litera>",
    "multi": "Dla każdej opcji napisz w osobnej linii: litera: TAK lub NIE, z bardzo krótkim powodem. W ostatniej linii napisz: Odpowiedź: <litery opcji z TAK, oddzielone przecinkami>",
    "tflist": "Oceń każde stwierdzenie osobno, dokładnie w brzmieniu z pytania, i go nie poprawiaj. W ostatniej linii napisz: Odpowiedź: <P lub F dla każdego stwierdzenia po kolei, oddzielone przecinkami>",
    "matching": "Dla każdego elementu napisz w osobnej linii: element: treść pasującej kategorii. W ostatniej linii napisz: Odpowiedź: element=litera; element=litera; ...",
    "order": "Dla każdej litery podaj rok lub wiek (np. A: 1410 r.). W ostatniej linii napisz: Odpowiedź: <litery od najwcześniejszego, oddzielone przecinkami>",
    "numeric": "Zapisz obliczenie w linii WYRAŻENIE: <działanie, np. 1525-1410>. W ostatniej linii napisz: Odpowiedź: <sama liczba>",
    "short": "Odpowiedz jak najkrócej, jednym do trzech słów. W ostatniej linii napisz: Odpowiedź: <odpowiedź>",
    "open": "Odpowiedz rzeczowo w 2-4 zdaniach, podając fakty, daty i nazwy.",
}

# v3: one line per item with the key fact first and the verdict last, so the model judges items
# one by one and the parser can read each verdict separately.
FORMAT_V3 = {
    "tflist": "Oceń każde stwierdzenie osobno i dokładnie w brzmieniu z pytania; nie poprawiaj go. Napisz po jednej linii na stwierdzenie: numer. kluczowy fakt => P albo F. W ostatniej linii napisz: Odpowiedź: <P lub F po kolei, oddzielone przecinkami>",
    "multi": "Sprawdź każdą opcję osobno. Napisz po jednej linii na opcję: litera: kluczowy fakt, najlepiej z datą => TAK albo NIE. W ostatniej linii napisz: Odpowiedź: <litery opcji z TAK, oddzielone przecinkami>",
    "matching": "Napisz po jednej linii na element: element => pełna treść pasującej kategorii. W ostatniej linii napisz: Odpowiedź: element=litera; element=litera; ...",
}
EXAMPLES_V3 = {
    "tflist": (
        "Oceń prawdziwość stwierdzeń. (1) Mikołaj Kopernik opisał teorię heliocentryczną w dziele „O obrotach sfer niebieskich”. (2) Reformację w 1517 r. zapoczątkował Jan Kalwin. Odpowiedz literami P/F w kolejności.",
        "1. Kopernik wydał „O obrotach sfer niebieskich” w 1543 r. => P\n2. W 1517 r. wystąpił Marcin Luter, nie Jan Kalwin => F\nOdpowiedź: P,F",
    ),
    "multi": (
        "Zaznacz wszystkie miasta, które były stolicami cesarstwa rzymskiego lub bizantyńskiego. (Może być kilka poprawnych, podaj wszystkie litery.)\nA) Rzym\nB) Konstantynopol\nC) Aleksandria\nD) Rawenna",
        "A: Rzym, stolica cesarstwa => TAK\nB: Konstantynopol, stolica od 330 r. => TAK\nC: Aleksandria nigdy nie była stolicą cesarstwa => NIE\nD: Rawenna, stolica zachodu od 402 r. => TAK\nOdpowiedź: A,B,D",
    ),
    "matching": (
        "Przyporządkuj postaciom epoki. Elementy: Perykles, Karol Wielki. Kategorie: A) średniowiecze; B) starożytność. Odpowiedz w formacie element=litera, np. Perykles=?; Karol Wielki=?.",
        "Perykles => starożytność (V w. p.n.e.)\nKarol Wielki => średniowiecze (VIII-IX w.)\nOdpowiedź: Perykles=B; Karol Wielki=A",
    ),
}
V3_MAX_TOKENS = 240  # one line per item needs more room than a bare answer

REASON = "Najpierw w 1-2 krótkich zdaniach przypomnij kluczowe fakty (daty, postacie, miejsca)."
NO_REASON = "Nie uzasadniaj. Napisz tylko linię z odpowiedzią."

EXAMPLES = {
    "single": (
        "Który władca założył Akademię Krakowską w 1364 r.?\nA) Kazimierz Wielki\nB) Bolesław Chrobry\nC) Władysław Łokietek\nD) Zygmunt Stary",
        "Akademię Krakowską założył w 1364 r. Kazimierz Wielki.\nOdpowiedź: A",
    ),
    "multi": (
        "Zaznacz wszystkie miasta, które były stolicami cesarstwa rzymskiego lub bizantyńskiego. (Może być kilka poprawnych, podaj wszystkie litery.)\nA) Rzym\nB) Konstantynopol\nC) Aleksandria\nD) Rawenna",
        "A: TAK (stolica cesarstwa)\nB: TAK (stolica od 330 r.)\nC: NIE (nie była stolicą cesarstwa)\nD: TAK (stolica zachodu od 402 r.)\nOdpowiedź: A,B,D",
    ),
    "tflist": (
        "Oceń prawdziwość stwierdzeń. (1) Mikołaj Kopernik opisał teorię heliocentryczną w dziele „O obrotach sfer niebieskich”. (2) Reformację w 1517 r. zapoczątkował Jan Kalwin. Odpowiedz literami P/F w kolejności.",
        "(1) Kopernik, dzieło z 1543 r.: prawda. (2) W 1517 r. wystąpił Marcin Luter, nie Kalwin: fałsz.\nOdpowiedź: P,F",
    ),
    "matching": (
        "Przyporządkuj postaciom epoki. Elementy: Perykles, Karol Wielki. Kategorie: A) średniowiecze; B) starożytność. Odpowiedz w formacie element=litera, np. Perykles=?; Karol Wielki=?.",
        "Perykles: starożytność (V w. p.n.e.)\nKarol Wielki: średniowiecze (VIII-IX w.)\nOdpowiedź: Perykles=B; Karol Wielki=A",
    ),
    "order": (
        "Uporządkuj wydarzenia chronologicznie. A) bitwa pod Cedynią; B) chrzest Mieszka I; C) zjazd gnieźnieński. Podaj litery w kolejności.",
        "A: 972 r.\nB: 966 r.\nC: 1000 r.\nOdpowiedź: B,A,C",
    ),
    "numeric": (
        "Oblicz, ile lat minęło od bitwy pod Grunwaldem (1410) do hołdu pruskiego (1525). Podaj samą liczbę.",
        "WYRAŻENIE: 1525-1410\nOdpowiedź: 115",
    ),
    "short": (
        "Podaj nazwę bitwy z 1605 r., w której wojska Rzeczypospolitej pod wodzą Jana Karola Chodkiewicza pokonały Szwedów. Odpowiedz jak najkrócej.",
        "Bitwa pod Kircholmem, 1605 r.\nOdpowiedź: Kircholm",
    ),
    "open": (
        "Wyjaśnij, dlaczego unię lubelską uznaje się za przełom w dziejach Polski i Litwy.",
        "Unia lubelska z 1569 r. zastąpiła unię personalną unią realną: powstała Rzeczpospolita Obojga Narodów ze wspólnym królem wybieranym wspólnie, wspólnym sejmem i polityką zagraniczną.",
    ),
}


def _with_style(qtype, answer):
    """Strip the reasoning line from an example when reasoning is switched off."""
    if config.REASONING or qtype in ("open", "numeric"):
        return answer
    return answer.splitlines()[-1]


_FORMAT_EXAMPLE = re.compile(r",?\s*np\.\s*(?:[A-H](?:\s*,\s*[A-H])+|[PF](?:\s*,\s*[PF])+)\.?")


def _without_format_examples(text):
    """Drop 'np. C,A,D,B' / 'np. P,F,P' from the question: small models copy them as the answer."""
    return _FORMAT_EXAMPLE.sub(".", text).replace("..", ".")


def _context_block(passages):
    if not passages:
        return ""
    lines = ["Fragmenty z encyklopedii (mogą pomóc, ale nie muszą dotyczyć pytania):"]
    for i, p in enumerate(passages, 1):
        lines.append(f"[{i}] {p['title']}: {p['text'][:config.PASSAGE_CHARS]}")
    return "\n".join(lines) + "\n\n"


def uses_item_lines(q):
    """v3 answers list questions one line per item, with a fact hint for every item."""
    return config.PROMPT_VERSION == "v3" and q.qtype in FORMAT_V3


def max_tokens(q):
    return max(config.MAX_TOKENS, V3_MAX_TOKENS) if uses_item_lines(q) else config.MAX_TOKENS


def _items(q):
    """(label, retrieval query) for each statement, option or element of a list question."""
    if q.qtype == "tflist":
        return [(f"{i}.", s, "") for i, s in enumerate(q.statements, 1)]
    if q.qtype == "multi":
        # The criterion lives in the stem ("w XIX wieku", "z dynastii Jagiellonów"): it breaks ties.
        return [(f"{l}:", t, q.stem) for l, t in q.options.items()]
    if q.qtype == "matching":
        return [(f"{e} =>", e, "") for e in q.elements]
    return []


def _hints_block(q):
    from .retrieval import item_hint  # imported here: retrieval loads the index lazily
    lines = [f"{label} {hint}" for label, text, context in _items(q)
             for hint in [item_hint(text, context)] if hint]
    if not lines:
        return ""
    return "Fakty do poszczególnych pozycji (z encyklopedii, mogą być niepełne):\n" + "\n".join(lines) + "\n\n"


def build_messages(q, passages):
    style = REASON if config.REASONING else NO_REASON
    if q.qtype in ("open",):
        style = ""
    instruction = f"{style} {FORMAT[q.qtype]}".strip()
    ex_q, ex_a = EXAMPLES[q.qtype]
    hints = ""
    if uses_item_lines(q):
        instruction = FORMAT_V3[q.qtype]
        ex_q, ex_a = EXAMPLES_V3[q.qtype]
        hints = _hints_block(q)
        passages = passages[:2]  # item hints take the room of the third passage
    image_note = "\n(Pytanie odnosi się do obrazu, którego nie widzisz. Wykorzystaj opis i wiedzę.)" if q.has_image else ""
    return [
        {"role": "system", "content": SYSTEM},
        *cke_shots(),
        {"role": "user", "content": f"Pytanie: {ex_q}\n\n{instruction}"},
        {"role": "assistant", "content": _with_style(q.qtype, ex_a)},
        {"role": "user", "content": f"{_context_block(passages)}{hints}Pytanie: {_without_format_examples(q.text)}{image_note}\n\n{instruction}"},
    ]


_CKE_SHOTS = None


def cke_shots():
    """v3: worked examples from real CKE papers, placed first and in a fixed order.

    They come from data/cke/fewshot.json, built locally by scripts/build_cke.py (CKE papers quote
    copyrighted sources, so they are not in the repository). Being an identical prefix for every
    question, they are read once and then served from the model server's prompt cache.
    """
    global _CKE_SHOTS
    if config.PROMPT_VERSION != "v3":
        return []
    if _CKE_SHOTS is None:
        _CKE_SHOTS = []
        path = config.CKE_DIR / "fewshot.json"
        if path.exists():
            for shot in json.loads(path.read_text(encoding="utf-8")):
                instruction = FORMAT_V3.get(shot["type"]) or f"{REASON} {FORMAT[shot['type']]}"
                _CKE_SHOTS += [
                    {"role": "user", "content": f"Pytanie: {_without_format_examples(shot['question'])}\n\n{instruction}"},
                    {"role": "assistant", "content": shot["answer"]},
                ]
        else:
            print(f"[prompts] v3 requested but {path} is missing; run scripts/build_cke.py")
    return _CKE_SHOTS
