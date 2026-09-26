"""Polish prompts: one instruction and one worked example per answer type.

Examples are deliberately different from the dev set questions, so the dev score is not inflated.
"""
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


def build_messages(q, passages):
    style = REASON if config.REASONING else NO_REASON
    if q.qtype in ("open",):
        style = ""
    instruction = f"{style} {FORMAT[q.qtype]}".strip()
    ex_q, ex_a = EXAMPLES[q.qtype]
    image_note = "\n(Pytanie odnosi się do obrazu, którego nie widzisz. Wykorzystaj opis i wiedzę.)" if q.has_image else ""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Pytanie: {ex_q}\n\n{instruction}"},
        {"role": "assistant", "content": _with_style(q.qtype, ex_a)},
        {"role": "user", "content": f"{_context_block(passages)}Pytanie: {_without_format_examples(q.text)}{image_note}\n\n{instruction}"},
    ]
