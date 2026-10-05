# Graph Report - voice_assistant_demo  (2026-10-05)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 160 nodes · 282 edges · 18 communities (9 shown, 9 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 16 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `6eb2d026`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- selfcheck.py
- spellcheck.py
- evaluate_model.py
- main.py
- Persian & English ASR Demo
- transcribe
- get_model
- pick_model_and_language
- set_active_model
- voice_assistant_demo
- run.sh
- convert_khosravi_large.sh
- convert_large_turbo.sh
- convert_model.sh
- fetch_amirmohseni.sh
- fetch_model.sh

## God Nodes (most connected - your core abstractions)
1. `run()` - 15 edges
2. `transcribe()` - 14 edges
3. `get_model()` - 11 edges
4. `spell_correct()` - 10 edges
5. `pick_model_and_language()` - 10 edges
6. `Persian & English ASR Demo` - 10 edges
7. `Things that will bite you` - 10 edges
8. `normalize()` - 9 edges
9. `evaluate()` - 9 edges
10. `main_()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `Things that will bite you` --references--> `closest_language()`  [INFERRED]
  AGENTS.md → backend/main.py
- `Things that will bite you` --references--> `_is_derived()`  [INFERRED]
  AGENTS.md → backend/spellcheck.py
- `Things that will bite you` --references--> `spell_correct()`  [INFERRED]
  AGENTS.md → backend/spellcheck.py
- `Things that will bite you` --references--> `wake_command()`  [INFERRED]
  AGENTS.md → backend/main.py
- `Code graph` --references--> `get_model()`  [INFERRED]
  README.md → backend/main.py

## Import Cycles
- None detected.

## Communities (18 total, 9 thin omitted)

### Community 0 - "selfcheck.py"
Cohesion: 0.08
Nodes (16): closest_language(), check_detection_routes_to_the_en_slot(), check_detector_can_actually_detect(), check_detector_is_not_the_fa_slot_by_accident(), check_fa_threshold(), check_fault_tolerance(), check_hotwords_param_picked(), check_language_is_closed() (+8 more)

### Community 1 - "spellcheck.py"
Cohesion: 0.12
Nodes (12): check_spell_correction_loads(), _candidate(), _correct_word(), demo(), _is_derived(), _load_vocabulary(), by_key(), normalize() (+4 more)

### Community 2 - "evaluate_model.py"
Cohesion: 0.15
Nodes (8): detect(), detect_report(), evaluate(), main_(), normalize(), spell_report(), transcribe(), word_errors()

### Community 3 - "main.py"
Cohesion: 0.20
Nodes (4): _discover_models(), health(), lifespan(), _resolved()

### Community 4 - "Persian & English ASR Demo"
Cohesion: 0.17
Nodes (11): API, Checks and evaluation, Configuration, Credits, Docker, Features, Model roles, Optional CUDA (+3 more)

### Community 5 - "transcribe"
Cohesion: 0.25
Nodes (3): save_clip(), transcribe(), wake_command()

### Community 6 - "get_model"
Cohesion: 0.39
Nodes (5): Things that will bite you, _build(), get_model(), reload_model(), with_fresh_retry()

### Community 8 - "set_active_model"
Cohesion: 0.33
Nodes (3): resolve_language(), set_active_model(), check_runtime_switches()

### Community 9 - "voice_assistant_demo"
Cohesion: 0.40
Nodes (4): Conventions, Layout, Running, voice_assistant_demo

## Knowledge Gaps
- **19 isolated node(s):** `LD_LIBRARY_PATH`, `run.sh script`, `convert_khosravi_large.sh script`, `convert_large_turbo.sh script`, `convert_model.sh script` (+14 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 79 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Code graph` connect `spellcheck.py` to `Persian & English ASR Demo`, `get_model`, `pick_model_and_language`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `get_model()` (e.g. with `Things that will bite you` and `Code graph`) actually correct?**
  _`get_model()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `LD_LIBRARY_PATH`, `run.sh script`, `convert_khosravi_large.sh script` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `selfcheck.py` be split into smaller, more focused modules?**
  _Cohesion score 0.08097165991902834 - nodes in this community are weakly interconnected._
- **Why does `Persian & English ASR Demo` connect `Persian & English ASR Demo` to `spellcheck.py`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `spell_correct()` (e.g. with `Things that will bite you` and `replace()`) actually correct?**
  _`spell_correct()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Should `spellcheck.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12333333333333334 - nodes in this community are weakly interconnected._