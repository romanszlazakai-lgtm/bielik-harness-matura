# Dev set

`historia_dev.jsonl`: 46 Polish history questions in the answer formats of the organisers' exam
sheet. Written by the team with LLM assistance during the hackathon; not taken from any exam paper.
The knowledge-base cheat sheet (`data/kb_extra/chronologia.md`) was written by the same team, so
scores on this set are optimistic. The organisers' practice exam is the real check.

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | question id |
| `type` | `single`, `multi`, `tflist`, `matching`, `order`, `numeric`, `short` |
| `question` | question text, including format instructions for matching, order and tflist |
| `options` | `{"A": "...", ...}` for single and multi (appended one per line when asked) |
| `answer` | canonical answer: `B`, `A,C,D`, `P,P,F`, `x=B; y=D`, `B,A,C,D`, `127`, `Wazowie` |
| `accept` | short answers only: accepted variants |

| Type | Items |
|---|---|
| single | 23 |
| multi | 5 |
| tflist | 4 |
| matching | 3 |
| order | 4 |
| short | 6 |
| numeric | 1 |
