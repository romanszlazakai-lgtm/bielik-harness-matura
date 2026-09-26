"""Minimal RTF to plain text (standard library only), enough for OCR exports such as FineReader's.

Skips font/colour/style tables, pictures and other destinations; decodes \\'hh bytes with the
document code page and \\uN Unicode escapes.
"""
import re

_SKIP = {"fonttbl", "colortbl", "stylesheet", "info", "pict", "object", "header", "footer",
         "headerl", "headerr", "footerl", "footerr", "listtable", "listoverridetable", "rsidtbl",
         "xmlnstbl", "themedata", "colorschememapping", "datastore", "latentstyles", "generator",
         "fldinst", "shppict", "nonshppict", "shpinst", "sp", "sn", "sv", "bkmkstart", "bkmkend",
         "pgdsctbl", "revtbl", "filetbl", "mmathPr", "wgrffmtfilter", "pnseclvl"}
_BREAKS = {"par": "\n", "line": "\n", "sect": "\n\n", "page": "\n\n", "row": "\n", "cell": " | ",
           "tab": "\t", "emdash": "-", "endash": "-", "bullet": "*", "lquote": "‘", "rquote": "’",
           "ldblquote": "“", "rdblquote": "”"}
_TOKEN = re.compile(r"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\([^a-zA-Z])|([{}])|([^\\{}\r\n]+)|[\r\n]+")


def rtf_to_text(rtf):
    m = re.search(r"\\ansicpg(\d+)", rtf[:2000])
    codepage = f"cp{m.group(1)}" if m else "cp1252"
    stack, out = [], []
    skip, uc, skip_chars, pending = False, 1, 0, bytearray()

    def flush_bytes():
        if pending:
            out.append(pending.decode(codepage, errors="replace"))
            pending.clear()

    for word, arg, hexbyte, sym, brace, text in _TOKEN.findall(rtf):
        if brace == "{":
            flush_bytes()
            stack.append((skip, uc))
        elif brace == "}":
            flush_bytes()
            skip, uc = stack.pop() if stack else (False, 1)
        elif hexbyte:
            if skip_chars:
                skip_chars -= 1
            elif not skip:
                pending.append(int(hexbyte, 16))
        elif skip:
            continue
        elif word:
            flush_bytes()
            if word in _SKIP:
                skip = True
            elif word == "uc":
                uc = int(arg or 1)
            elif word == "u":
                code = int(arg)
                out.append(chr(code + 65536 if code < 0 else code))
                skip_chars = uc
            elif word in _BREAKS:
                out.append(_BREAKS[word])
        elif sym:
            flush_bytes()
            if sym == "*":
                skip = True
            elif sym in "\\{}":
                out.append(sym)
            elif sym == "~":
                out.append(" ")
            elif sym in "-_":
                out.append("-" if sym == "_" else "")
        elif text:
            flush_bytes()
            if skip_chars:
                drop = min(skip_chars, len(text))
                text, skip_chars = text[drop:], skip_chars - drop
            out.append(text)
    flush_bytes()
    result = "".join(out)
    result = re.sub(r"[ \t]+\n", "\n", result)
    return re.sub(r"\n{3,}", "\n\n", result).strip()
