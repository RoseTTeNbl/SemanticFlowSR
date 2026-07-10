# Task Plan: SemanticFlowSR History Cleanup

## Goal
Understand the current SemanticFlowSR theory, implementation, and experiment state, then remove or quarantine historical-version traces so docs, code entrypoints, logs, and result directories present a clean current mainline.

## Current Phase
Complete

## Phases

### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints
- [x] Document algorithm and experiment state in findings.md
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Define what is current mainline versus legacy/archive
- [x] Define cleanup actions for docs, logs, results, and scripts/code
- **Status:** complete

### Phase 3: Implementation
- [x] Clean or archive obsolete docs/log/result/code traces without deleting current evidence
- [x] Keep README/AGENTS/results/logs references aligned
- **Status:** complete

### Phase 4: Testing & Verification
- [x] Run lightweight static/listing checks
- [x] Run targeted tests if cleanup touches executable code
- [x] Document verification results
- **Status:** complete

### Phase 5: Delivery
- [x] Review outputs
- [x] Deliver to user
- **Status:** complete

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Do not launch long GPU training during cleanup | AGENTS.md says training has been stopped during cleanup unless explicitly requested. |
| Treat token construction, semantic endpoint correction, online semantic mass guidance, target-conditioned Stage1 runners, and semantic latent endpoint as legacy/failed probes by default | Current algorithm/math/architecture docs and code now define One-Step Semantic Fisher Cycle as the only supported mainline. |
| Keep `one_step_semantic_fisher_cycle_cpu_two_iter_20260710` as the visible current result | `docs/STRUCTURAL_CLOSURE.md` names it as the retained two-iteration evidence for the current loop. |
| Move failed 20260710 probes under `_legacy_failed_20260710` and remove obsolete 20260709 semantic-mass manifests | This keeps evidence reachable without mixing it into current-result listings. |

## Errors Encountered
| Error | Resolution |
|-------|------------|
| Root `/home/ywj/wyh` is not a git repository | Switched git inspection to `/home/ywj/wyh/SFSR/SemanticFlowSR`. |
