# Chess Position Reasoning Trace Generator

## Complete Technical Design Document

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [System Architecture](#system-architecture)
4. [Data Collection Modules](#data-collection-modules)
5. [Trace Structure](#trace-structure)
6. [Reasoning Patterns](#reasoning-patterns)
7. [Template System](#template-system)
8. [Variety Mechanisms](#variety-mechanisms)
9. [Implementation Details](#implementation-details)
10. [Example Outputs](#example-outputs)
11. [File Structure](#file-structure)
12. [Dependencies](#dependencies)
13. [Further Enrichment Ideas](#further-enrichment-ideas)

---

## Overview

This system generates **human-like reasoning traces** for chess positions. Given a FEN string, it produces a text explanation that reads like a player thinking through the position—exploring candidates, considering threats, eliminating bad moves, and arriving at a conclusion.

### Key Principles

- **UCI notation throughout** - All moves in Universal Chess Interface format (e.g., `e2e4`, `g1f3`)
- **Discovery-oriented** - We explore and reason; we don't declare "best" until the end
- **Text-focused** - Minimal numbers; descriptive language over evaluations
- **Randomized exploration** - Candidate moves analyzed in shuffled order
- **Human-like thinking** - Questions, uncertainty, reconsideration, intuition

---

## Design Philosophy

### Why This Matters

Traditional chess engine output looks like:
```
info depth 20 score cp 45 pv e2e4 e7e5 g1f3 b8c6
bestmove e2e4
```

This is **not** how humans think about chess. We want output like:

> "Looking at this position, the king on g8 seems safe for now, but I notice the f7 square is only defended by the king itself. Let me consider the candidate moves...
> 
> The move d2d4 pushes forward in the center. This would challenge Black's control, but it does leave the d4 pawn without immediate support. If Black responds with e5xd4, we'd need to recapture...
> 
> What about g1f3? This develops the knight toward the center and prepares castling. It also eyes the e5 pawn. The knight on f3 would coordinate well with a future bishop on c4...
> 
> [continues exploring]
> 
> After considering these ideas, g1f3 stands out. It develops a piece, prepares kingside castling, and maintains flexibility. This is the move I'd play here."

### Core Tenets

1. **Explore before concluding** - Never say "best" until the final synthesis
2. **Show the journey** - Include dead ends, reconsiderations, comparisons
3. **Think in concepts** - "Weak square", "open file", "king safety" not "+0.45"
4. **Ask questions** - "What is the threat?", "Can we exploit this?"
5. **Use uncertainty** - "This seems strong", "I'm not entirely sure, but..."
6. **Vary the approach** - Sometimes thorough, sometimes brief, sometimes focused

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REASONING TRACE GENERATOR                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   INPUT: FEN String                                                          │
│      │                                                                       │
│      ▼                                                                       │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                    ANALYSIS ORCHESTRATOR                            │    │
│   │  • Coordinates all data collection                                  │    │
│   │  • Manages engine lifecycle                                         │    │
│   │  • Randomizes candidate order                                       │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│          │                                                                   │
│          ├──────────────────────────────────────────────────────────────┐   │
│          ▼                                                              │   │
│   ┌─────────────────────────────────────────────────────────────────┐  │   │
│   │                     DATA COLLECTORS                              │  │   │
│   ├─────────┬─────────┬─────────┬─────────┬─────────┬───────────────┤  │   │
│   │ MultiPV │  NNUE   │ Pattern │ Opening │Tablebase│    Trap       │  │   │
│   │  + WDL  │  Eval   │Detection│ Lookup  │  Probe  │  Detector     │  │   │
│   │         │Breakdown│         │  (ECO)  │(Syzygy) │               │  │   │
│   └─────────┴─────────┴─────────┴─────────┴─────────┴───────────────┘  │   │
│          │                                                              │   │
│          ▼                                                              │   │
│   ┌─────────────────────────────────────────────────────────────────┐  │   │
│   │                    ANALYSIS BUNDLE                               │  │   │
│   │  Unified data structure with all collected information           │  │   │
│   └─────────────────────────────────────────────────────────────────┘  │   │
│          │                                                              │   │
│          ▼                                                              │   │
│   ┌─────────────────────────────────────────────────────────────────┐  │   │
│   │                   TRACE SYNTHESIZER                              │  │   │
│   │  • Selects trace style and structure                            │  │   │
│   │  • Randomizes candidate exploration order                        │  │   │
│   │  • Generates human-like reasoning text                          │  │   │
│   │  • Builds toward conclusion                                      │  │   │
│   └─────────────────────────────────────────────────────────────────┘  │   │
│          │                                                              │   │
│          ▼                                                              │   │
│   OUTPUT: Reasoning Trace (string)                                      │   │
│                                                                         │   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Collection Modules

### Module 1: MultiPV Analysis + WDL

**Purpose**: Get top-K candidate moves with win/draw/loss probabilities.

**Key Data Extracted**:
```python
@dataclass
class PVLine:
    rank: int                      # Original engine ranking
    move_uci: str                  # Move in UCI (e.g., "e2e4")
    pv_uci: List[str]              # Full line in UCI
    wdl: Tuple[int, int, int]      # (win, draw, loss) per mille
    depth: int
    is_mate: bool
    mate_in: Optional[int]
    
    # Derived
    win_percent: float             # wdl[0] / 10
    draw_percent: float            # wdl[1] / 10  
    loss_percent: float            # wdl[2] / 10
```

**WDL Interpretation for Text**:
| Win % | Text Description |
|-------|------------------|
| 80%+  | "winning", "decisive advantage" |
| 60-80% | "clearly better", "significant edge" |
| 45-60% | "slightly better", "small advantage" |
| 40-55% | "roughly equal", "balanced" |
| 20-40% | "slightly worse", "under pressure" |
| <20%  | "losing", "serious trouble" |

---

### Module 2: NNUE Eval Breakdown

**Purpose**: Understand *why* a position is good/bad by examining per-piece contributions.

**Key Data Extracted**:
```python
@dataclass  
class PieceContribution:
    square: str                    # e.g., "e4"
    piece_type: str                # e.g., "knight", "bishop"
    piece_color: str               # "white" or "black"
    contribution: str              # "strong", "well-placed", "weak", "misplaced"
    
@dataclass
class EvalInsights:
    best_piece: Optional[PieceContribution]
    worst_piece: Optional[PieceContribution]
    white_activity: str            # "active", "passive", "cramped"
    black_activity: str
    key_insight: str               # Generated text explanation
```

**Contribution Thresholds** (in NNUE piece value):
| Value | Interpretation |
|-------|----------------|
| > +2.0 | "excellently placed", "dominating" |
| +1.0 to +2.0 | "well-placed", "active" |
| +0.5 to +1.0 | "reasonable", "decent" |
| -0.5 to +0.5 | "neutral" |
| -1.0 to -0.5 | "slightly misplaced" |
| < -1.0 | "poorly placed", "inactive" |

---

### Module 3: Pattern Detection

**Purpose**: Identify tactical and positional motifs.

**Patterns Detected**:

#### Tactical Motifs
| Pattern | Detection Logic |
|---------|-----------------|
| Fork | Piece attacks 2+ enemy pieces of value ≥3 |
| Pin | Sliding piece attacks through a piece to a more valuable piece/king |
| Skewer | Sliding piece attacks through more valuable piece to less valuable |
| Discovered Attack | Moving piece reveals attack from behind |
| Double Check | Two pieces give check simultaneously |
| Back Rank Threat | King trapped on back rank with mate threat |
| Hanging Piece | Undefended piece under attack |
| Trapped Piece | Piece with no safe squares, under attack |
| Overloaded Defender | Piece defending multiple threats |
| Deflection | Capture that removes a key defender |
| X-ray Attack | Slider pressures a high-value piece through a blocker |
| Decoy | Check that lures the king onto a vulnerable square |
| Sacrifice | Forcing move that leaves a piece en prise |
| Zwischenzug | Forcing intermezzo while a piece is hanging |
| Stalemate Trick | Move that forces immediate stalemate |
| Perpetual Check Idea | Multiple checks in the PV suggest a perpetual |
| Quiet Move | Non-capture, non-check improving move |

#### Positional Motifs
| Pattern | Detection Logic |
|---------|-----------------|
| Weak Square | Square not defended by pawns, accessible by enemy |
| Open File | File with no pawns |
| Half-Open File | File with only enemy pawns |
| Outpost | Protected square in enemy territory |
| Backward Pawn | Pawn that cannot be defended by other pawns |
| Isolated Pawn | Pawn with no adjacent pawns |
| Doubled Pawns | Two pawns on same file |
| Passed Pawn | Pawn with no enemy pawns blocking or attacking advance |
| Exposed King | King with insufficient pawn cover |
| Bishop Pair | Both bishops vs knight+bishop or two knights |

#### Move-Specific Patterns
| Pattern | Detection Logic |
|---------|-----------------|
| Check | Move gives check |
| Capture | Move captures piece |
| Promotion | Pawn reaches 8th rank |
| Castling | King castles |
| En Passant | Special pawn capture |
| Sacrifice | Giving up material for compensation |
| Quiet Move | Non-capture, non-check move |
| Zwischenzug | Intermediate move before expected response |

---

### Module 4: Opening Lookup

**Purpose**: Identify opening name and ECO code.

**Data Source**: lichess-org/chess-openings TSV files

**Key Data**:
```python
@dataclass
class OpeningInfo:
    eco: str           # "B90"
    name: str          # "Sicilian Defense: Najdorf Variation"
    main_line: str     # "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4 f3d4 g8f6 b1c3 a7a6"
```

---

### Module 5: Tablebase Probing

**Purpose**: Perfect endgame information for positions with ≤7 pieces.

**Key Data**:
```python
@dataclass
class TablebaseInfo:
    result: str           # "winning", "drawing", "losing"
    moves_to_end: int     # DTZ (distance to zeroing)
    is_available: bool    # Whether tablebase covers this position
    piece_count: int
```

---

### Module 6: Trap Detection

**Purpose**: Find moves that look good but are actually bad (or vice versa).

**Method**: Compare shallow (depth 8) vs deep (depth 20) analysis.

**Key Data**:
```python
@dataclass
class TrapInfo:
    move_uci: str
    appears: str          # "promising", "tempting", "natural"
    actually: str         # "loses material", "misses the win", "allows counterplay"
    shallow_assessment: str  # "looks good"
    deep_assessment: str     # "actually problematic"
    refutation_line: List[str]  # How opponent punishes
```

Optional detail: if per-candidate PVs are available, the generator can emit a short
refutation line (UCI by default) using the config key `trap_refutation_max_len`.

---

## Trace Structure

### The Journey Pattern

Every trace follows a journey from **uncertainty to conclusion**:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. ORIENTATION                                                  │
│     What kind of position is this? What stands out?             │
│     Opening context if applicable                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. ASSESSMENT                                                   │
│     Who stands better? Any immediate threats?                   │
│     Key features: king safety, piece activity, pawn structure   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. CANDIDATE GENERATION                                         │
│     What moves deserve consideration?                           │
│     List 3-5 candidates with brief first impressions            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. EXPLORATION (RANDOMIZED ORDER)                               │
│     Analyze each candidate in shuffled order                    │
│     For each: idea, consequences, problems, counter-ideas       │
│     Include some dead ends and reconsiderations                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. COMPARISON                                                   │
│     How do the candidates compare?                              │
│     Trade-offs, different character of positions                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. CONCLUSION                                                   │
│     Based on this analysis, [MOVE] is the choice                │
│     Final PV and summary of key reasoning                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reasoning Patterns

### Thinking Phrases

These make the trace feel like genuine exploration:

#### Exploration Starters
```
"Let me look at this position..."
"What's going on here?"
"First, I need to understand..."
"Looking at the board..."
"The position after this move..."
"Starting with the basics..."
```

#### Candidate Introduction
```
"A few moves catch my eye..."
"The candidates that stand out are..."
"I should consider..."
"Let me think about..."
"What about...?"
"There's also..."
```

#### Analysis Phrases
```
"If I play [move]..."
"The idea behind [move] is..."
"This threatens..."
"The problem with this is..."
"But wait, what about...?"
"Let me check if..."
"I notice that..."
"This would allow..."
```

#### Uncertainty Expressions
```
"I'm not entirely sure, but..."
"This seems to..."
"My instinct says..."
"This feels like..."
"It looks like..."
"Probably..."
"I think..."
```

#### Reconsideration
```
"Actually, looking again..."
"Wait, I missed..."
"Hmm, but..."
"On second thought..."
"That changes things..."
"I need to reconsider..."
```

#### Dead Ends
```
"This doesn't work because..."
"I looked at [move] but it fails to..."
"Unfortunately, this runs into..."
"This would be nice, but..."
"The problem is..."
```

#### Comparison Phrases
```
"Compared to [other move]..."
"Unlike [other move], this..."
"The difference is..."
"While [move1] does X, [move2] achieves Y..."
"Both moves have merit, but..."
```

#### Conclusion Phrases
```
"After considering everything..."
"Based on this analysis..."
"Taking all this into account..."
"The move that makes the most sense is..."
"Given these considerations, I'd play..."
"This leads me to choose..."
```

---

### Tactical Description Patterns

Instead of numbers, describe tactics in text:

#### Fork Descriptions
```
"The knight on [square] attacks both the [piece1] and [piece2] simultaneously"
"This creates a double attack—the [piece] can't save everything"
"The [piece] lands with a fork, hitting [target1] and [target2]"
```

#### Pin Descriptions
```
"The [piece1] is pinned to the [piece2] behind it"
"The [defender] can't move without exposing the [more valuable piece]"
"There's an uncomfortable pin on the [file/diagonal]"
```

#### King Safety Descriptions
```
"The king looks a bit exposed with those pawns pushed forward"
"Castling would tuck the king away to safety"
"The king is stuck in the center, which could be dangerous"
"There's a potential back rank issue here"
```

#### Piece Activity Descriptions
```
"The knight is beautifully centralized on [square]"
"That bishop is stuck behind its own pawns"
"The rooks haven't connected yet"
"This piece doesn't have much to do at the moment"
"The [piece] is working overtime, defending multiple things"
```

---

### Positional Description Patterns

#### Pawn Structure
```
"The pawns are somewhat compromised on the [side]"
"There's an isolated pawn on [file] that could become weak"
"The doubled pawns on [file] aren't ideal"
"The pawn chain points toward the [side]"
"That passed pawn could become dangerous"
```

#### Space and Control
```
"White has more space in the center"
"The [piece] controls the [diagonal/file]"
"There's a nice outpost on [square]"
"The [square] is weak and could be occupied"
```

#### Development
```
"Development is nearly complete"
"A few pieces still need to come out"
"The [piece] hasn't found a good square yet"
"Both sides are equally developed"
```

---

## Template System

### Trace Templates

Templates provide structural variety. Each template defines:
1. Which sections to include
2. Level of detail
3. Tone and style
4. Conclusion format

#### Template: Thorough Analysis

```
[ORIENTATION]
{opening_context_if_available}
{position_type_description}

[ASSESSMENT]
{who_stands_better}
{key_features: king_safety, piece_activity, pawn_structure}
{any_immediate_threats}

[CANDIDATES]
{intro_phrase}
{list_candidates_with_first_impressions}

[EXPLORATION - RANDOMIZED]
{for_each_candidate_shuffled:}
  {analysis_intro_phrase}
  {idea_behind_move}
  {concrete_variation}
  {problems_or_benefits}
  {tactical_patterns_if_any}
  {maybe_reconsideration}

[SYNTHESIS]
{comparison_of_candidates}
{elimination_of_inferior_moves}

[CONCLUSION]
{conclusion_phrase} {best_move_uci}
{final_pv}
{key_reason_summary}
```

#### Template: Quick Assessment

```
[ORIENTATION]
{brief_position_description}

[CANDIDATES]
{quick_list_of_moves}

[EXPLORATION - ABBREVIATED]
{for_top_3_candidates:}
  {move}: {one_line_idea}. {quick_variation}

[CONCLUSION]
{conclusion_phrase} {best_move_uci}
{short_justification}
```

#### Template: Problem-Focused

```
[PROBLEM IDENTIFICATION]
{what_challenge_does_position_present}
{what_does_opponent_threaten}

[ATTEMPTS]
{first_attempt}: {why_it_fails}
{second_attempt}: {why_it_fails}
{successful_attempt}: {why_it_works}

[CONCLUSION]
{the_move}: {best_move_uci}
{how_it_solves_the_problem}
{resulting_variation}
```

#### Template: Intuition + Verification

```
[FIRST IMPRESSION]
{gut_feeling_about_position}
{initial_move_preference}

[VERIFICATION]
{checking_the_intuition}
{what_concrete_lines_show}
{any_surprises}

[CONCLUSION]
{confirmation_or_adjustment}
{final_choice}: {best_move_uci}
```

#### Template: Comparison-Focused

```
[THE QUESTION]
{frame_main_decision}
{two_or_three_main_candidates}

[SIDE BY SIDE]
{candidate_1}: {pros} / {cons}
{candidate_2}: {pros} / {cons}
{candidate_3_if_applicable}: {pros} / {cons}

[THE DIFFERENCE]
{what_makes_one_better}
{key_distinguishing_factor}

[CONCLUSION]
{choice}: {best_move_uci}
{reasoning_summary}
```

---

### Section Templates

#### Orientation Templates

```python
ORIENTATION_TEMPLATES = [
    "This is a {opening_name} position. {characteristic_feature}.",
    
    "Looking at the board, {first_observation}. {second_observation}.",
    
    "We're in a {game_phase} with {piece_balance}. {notable_feature}.",
    
    "The position arose from {opening_name}. {current_situation}.",
    
    "{side_to_move} to move. {position_character}.",
    
    "An interesting position. {what_stands_out}.",
]
```

#### Candidate Introduction Templates

```python
CANDIDATE_INTRO_TEMPLATES = [
    "Several moves deserve attention here...",
    
    "Let me consider the candidates: {move_list}.",
    
    "I see a few options worth exploring.",
    
    "The moves that catch my eye are {move_list}.",
    
    "What should I look at? {move1} seems natural, and there's also {move2}, {move3}...",
    
    "A few ideas come to mind.",
]
```

#### Move Analysis Templates

```python
MOVE_ANALYSIS_TEMPLATES = [
    "{move} {idea_phrase}. {consequence}. {evaluation_phrase}.",
    
    "What about {move}? {idea_description}. If {response}, then {continuation}.",
    
    "The move {move} is interesting because {reason}. The line might go {pv}.",
    
    "{move}: {brief_idea}. {problem_or_benefit}.",
    
    "Let me check {move}. {analysis}. {conclusion_for_this_move}.",
    
    "Playing {move} would {effect}. {follow_up}.",
]
```

#### Conclusion Templates

```python
CONCLUSION_TEMPLATES = [
    "After considering these options, {move} stands out as the strongest choice. {key_reason}. The main line continues: {pv}.",
    
    "Based on this analysis, I would play {move}. {summary}. Expected play: {pv}.",
    
    "Taking everything into account, {move} is the move. {reasoning}. Line: {pv}.",
    
    "The analysis points to {move}. {why}. Play could continue {pv}.",
    
    "{move} is the answer here. {brief_justification}. Variation: {pv}.",
    
    "Having explored the alternatives, {move} makes the most sense. {conclusion_reasoning}. PV: {pv}.",
]
```

---

## Variety Mechanisms

### Randomization Points

1. **Template Selection**: Randomly choose from available templates weighted by position type
2. **Candidate Order**: Always shuffle the order in which candidates are explored
3. **Phrase Selection**: Randomly select from phrase pools for each section
4. **Detail Level**: Randomly vary depth of analysis for each candidate
5. **Optional Sections**: Probabilistically include/exclude certain observations
6. **Reconsideration Insertion**: Sometimes add "wait, actually..." moments
7. **Dead End Inclusion**: Sometimes show a move that was considered but rejected

### Variety Parameters

```python
@dataclass
class VarietyConfig:
    # Template weights
    thorough_weight: float = 0.30
    quick_weight: float = 0.25
    problem_focused_weight: float = 0.15
    intuition_weight: float = 0.15
    comparison_weight: float = 0.15
    
    # Section probabilities
    include_opening_context: float = 0.7
    include_piece_activity_comment: float = 0.6
    include_king_safety_comment: float = 0.5
    include_pawn_structure_comment: float = 0.4
    include_trap_warning: float = 0.8  # when trap exists
    include_dead_end: float = 0.3
    include_reconsideration: float = 0.25
    
    # Detail variation
    min_candidates_explored: int = 2
    max_candidates_explored: int = 5
    min_pv_length: int = 2
    max_pv_length: int = 6
    
    # Phrase variety
    use_questions: float = 0.4
    use_uncertainty: float = 0.3
    use_thinking_aloud: float = 0.5
```

### Position-Dependent Adjustments

```python
def adjust_for_position(config: VarietyConfig, analysis: AnalysisBundle) -> VarietyConfig:
    """Adjust variety parameters based on position characteristics"""
    
    # Tactical positions: more thorough, more concrete
    if len(analysis.patterns) > 3:
        config.thorough_weight += 0.2
        config.include_dead_end += 0.2
    
    # Quiet positions: more positional commentary
    if analysis.is_quiet_position:
        config.include_pawn_structure_comment += 0.3
        config.include_piece_activity_comment += 0.2
    
    # Endgames: more concrete, less verbose
    if analysis.tablebase or analysis.is_endgame:
        config.quick_weight += 0.3
        config.max_candidates_explored = 3
    
    # When there are traps: educational approach
    if analysis.traps and analysis.traps.has_traps:
        config.include_trap_warning = 0.95
        config.problem_focused_weight += 0.2
    
    # Opening positions: include theory context
    if analysis.opening:
        config.include_opening_context = 0.9
    
    return config
```

---

## Implementation Details

### Core Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import random

# All moves in UCI format
UCIMove = str  # e.g., "e2e4", "g1f3", "e7e8q"

@dataclass
class CandidateMove:
    """A move being considered"""
    move_uci: UCIMove
    pv_uci: List[UCIMove]
    
    # Win/Draw/Loss as percentages (0-100)
    win_percent: float
    draw_percent: float
    loss_percent: float
    
    # Qualitative assessment
    assessment: str  # "winning", "better", "equal", "worse", "losing"
    
    # Tactical elements
    is_check: bool
    is_capture: bool
    is_promotion: bool
    captured_piece: Optional[str]  # "queen", "rook", etc.
    
    # Patterns involved
    patterns: List[str]  # ["fork", "pin", etc.]
    
    # For trap detection
    is_trap: bool = False
    trap_type: Optional[str] = None  # "tempting_blunder", "hidden_resource"
    trap_description: Optional[str] = None

@dataclass
class PositionContext:
    """Context about the position"""
    fen: str
    side_to_move: str  # "white" or "black"
    
    # Opening info
    opening_name: Optional[str] = None
    opening_eco: Optional[str] = None
    
    # Game phase
    phase: str = "middlegame"  # "opening", "middlegame", "endgame"
    
    # Overall assessment  
    who_is_better: str = "equal"  # "white", "black", "equal"
    assessment_reason: str = ""
    
    # Key features
    white_king_safety: str = "safe"  # "safe", "slightly exposed", "exposed", "in danger"
    black_king_safety: str = "safe"
    piece_activity: Dict[str, str] = field(default_factory=dict)  # piece -> assessment
    pawn_structure_notes: List[str] = field(default_factory=list)
    
    # Threats
    immediate_threats: List[str] = field(default_factory=list)
    
    # Tablebase
    tablebase_result: Optional[str] = None
    tablebase_dtz: Optional[int] = None

@dataclass
class AnalysisBundle:
    """Complete analysis of a position"""
    context: PositionContext
    candidates: List[CandidateMove]  # Will be shuffled for exploration
    patterns: List[str]
    
    # The actual best move (only revealed at end)
    best_move_uci: UCIMove
    best_pv_uci: List[UCIMove]
    
    # For variety
    exploration_order: List[int] = field(default_factory=list)  # Shuffled indices
```

### Main Generator Class

```python
class ReasoningTraceGenerator:
    """
    Generates human-like reasoning traces for chess positions.
    
    Usage:
        config = GeneratorConfig(stockfish_path="/usr/bin/stockfish")
        generator = ReasoningTraceGenerator(config)
        
        trace = generator.generate("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        print(trace)
    """
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.analyzer = None  # Lazy initialization
        self.synthesizer = TraceSynthesizer()
        self.variety = VarietyConfig()
        
    def generate(self, fen: str) -> str:
        """Generate a reasoning trace for the position"""
        
        # 1. Collect all analysis data
        analysis = self._analyze(fen)
        
        # 2. Shuffle candidate exploration order
        analysis.exploration_order = list(range(len(analysis.candidates)))
        random.shuffle(analysis.exploration_order)
        
        # 3. Adjust variety based on position
        variety = adjust_for_position(self.variety, analysis)
        
        # 4. Select template
        template = self._select_template(analysis, variety)
        
        # 5. Generate trace
        return self.synthesizer.generate(analysis, template, variety)
    
    def _analyze(self, fen: str) -> AnalysisBundle:
        """Collect all analysis data"""
        # ... implementation
        pass
    
    def _select_template(self, analysis: AnalysisBundle, variety: VarietyConfig) -> str:
        """Select which template to use"""
        weights = {
            "thorough": variety.thorough_weight,
            "quick": variety.quick_weight,
            "problem_focused": variety.problem_focused_weight,
            "intuition": variety.intuition_weight,
            "comparison": variety.comparison_weight,
        }
        
        # Normalize and select
        total = sum(weights.values())
        r = random.random() * total
        cumulative = 0
        
        for template, weight in weights.items():
            cumulative += weight
            if r <= cumulative:
                return template
        
        return "thorough"
```

### Trace Synthesizer

```python
class TraceSynthesizer:
    """Generates the actual trace text"""
    
    def __init__(self):
        self.phrases = PhraseLibrary()
        
    def generate(
        self, 
        analysis: AnalysisBundle, 
        template: str,
        variety: VarietyConfig
    ) -> str:
        """Generate complete trace"""
        
        sections = []
        
        # 1. Orientation
        sections.append(self._orientation(analysis, variety))
        
        # 2. Assessment (sometimes)
        if random.random() < 0.7:
            sections.append(self._assessment(analysis, variety))
        
        # 3. Candidates
        sections.append(self._candidates(analysis, variety))
        
        # 4. Exploration (in shuffled order!)
        sections.append(self._exploration(analysis, variety))
        
        # 5. Comparison/Synthesis (sometimes)
        if template in ["thorough", "comparison"] and random.random() < 0.6:
            sections.append(self._comparison(analysis, variety))
        
        # 6. Conclusion (always - this is where we reveal the best move)
        sections.append(self._conclusion(analysis, variety))
        
        return "\n\n".join(sections)
    
    def _orientation(self, analysis: AnalysisBundle, variety: VarietyConfig) -> str:
        """Generate orientation section"""
        parts = []
        
        ctx = analysis.context
        
        # Opening context
        if ctx.opening_name and random.random() < variety.include_opening_context:
            parts.append(random.choice([
                f"This position comes from the {ctx.opening_name}.",
                f"We're in a {ctx.opening_name} ({ctx.opening_eco}).",
                f"This is the {ctx.opening_name}.",
            ]))
        
        # Phase/character
        phase_descriptions = {
            "opening": ["Still in the opening phase.", "Development isn't complete yet."],
            "middlegame": ["A typical middlegame position.", "We're in the thick of the middlegame."],
            "endgame": ["This is an endgame.", "We've reached the endgame."],
        }
        if random.random() < 0.5:
            parts.append(random.choice(phase_descriptions.get(ctx.phase, [""])))
        
        # Side to move
        parts.append(f"{ctx.side_to_move.title()} to move.")
        
        return " ".join(p for p in parts if p)
    
    def _assessment(self, analysis: AnalysisBundle, variety: VarietyConfig) -> str:
        """Generate position assessment"""
        parts = []
        ctx = analysis.context
        
        # Who is better
        assessment_phrases = {
            "white_winning": [
                "White has a decisive advantage here.",
                "White is winning.",
            ],
            "white_better": [
                "White is clearly better.",
                "White has the upper hand.",
            ],
            "white_slight": [
                "White is slightly better.",
                "White has a small edge.",
            ],
            "equal": [
                "The position is roughly equal.",
                "Neither side has a significant advantage.",
                "It's pretty balanced.",
            ],
            "black_slight": [
                "Black is slightly better.",
                "Black has a small edge.",
            ],
            "black_better": [
                "Black is clearly better.",
                "Black has the upper hand.",
            ],
            "black_winning": [
                "Black has a decisive advantage.",
                "Black is winning.",
            ],
        }
        
        parts.append(random.choice(assessment_phrases.get(ctx.who_is_better, assessment_phrases["equal"])))
        
        # King safety (sometimes)
        if random.random() < variety.include_king_safety_comment:
            if ctx.white_king_safety in ["exposed", "in danger"]:
                parts.append("The white king looks vulnerable.")
            if ctx.black_king_safety in ["exposed", "in danger"]:
                parts.append("The black king is somewhat exposed.")
        
        # Immediate threats
        if ctx.immediate_threats:
            parts.append(f"There's a threat of {ctx.immediate_threats[0]}.")
        
        return " ".join(parts)
    
    def _candidates(self, analysis: AnalysisBundle, variety: VarietyConfig) -> str:
        """Generate candidate moves section"""
        
        intro = random.choice([
            "Let me consider the candidate moves.",
            "A few moves deserve attention here.",
            "What options do we have?",
            "I should look at several possibilities.",
            "The candidates worth considering:",
        ])
        
        # List candidates with brief first impressions
        candidate_lines = []
        for cand in analysis.candidates[:variety.max_candidates_explored]:
            impression = self._first_impression(cand)
            candidate_lines.append(f"- {cand.move_uci}: {impression}")
        
        return intro + "\n" + "\n".join(candidate_lines)
    
    def _first_impression(self, cand: CandidateMove) -> str:
        """Generate brief first impression of a candidate"""
        
        if cand.is_check:
            return random.choice(["gives check", "checks the king", "a check"])
        
        if cand.is_capture:
            return random.choice([
                f"captures the {cand.captured_piece}",
                f"takes on that square",
                f"a capture",
            ])
        
        if cand.patterns:
            pattern = cand.patterns[0]
            return random.choice([
                f"creates a {pattern} threat",
                f"looks like a {pattern} idea",
                f"sets up {pattern}",
            ])
        
        impressions = [
            "looks interesting",
            "worth checking",
            "a natural move",
            "develops/improves a piece",
            "fights for the center",
            "prepares something",
        ]
        return random.choice(impressions)
    
    def _exploration(self, analysis: AnalysisBundle, variety: VarietyConfig) -> str:
        """Generate exploration of candidates IN SHUFFLED ORDER"""
        
        parts = []
        
        # Use the shuffled exploration order!
        for idx in analysis.exploration_order[:variety.max_candidates_explored]:
            cand = analysis.candidates[idx]
            
            # Vary the depth of analysis
            if random.random() < 0.3:
                # Brief analysis
                parts.append(self._brief_analysis(cand, analysis))
            else:
                # Detailed analysis
                parts.append(self._detailed_analysis(cand, analysis, variety))
            
            # Sometimes add reconsideration
            if random.random() < variety.include_reconsideration:
                parts.append(self._reconsideration(cand, analysis))
        
        # Maybe add a dead end
        if random.random() < variety.include_dead_end:
            parts.append(self._dead_end(analysis))
        
        return "\n\n".join(parts)
    
    def _brief_analysis(self, cand: CandidateMove, analysis: AnalysisBundle) -> str:
        """Short analysis of a candidate"""
        
        move = cand.move_uci
        
        templates = [
            f"{move} — {self._move_idea(cand)}. {self._quick_assessment(cand)}",
            f"The move {move} {self._move_effect(cand)}.",
            f"Looking at {move}: {self._brief_line(cand)}",
        ]
        
        return random.choice(templates)
    
    def _detailed_analysis(
        self, 
        cand: CandidateMove, 
        analysis: AnalysisBundle,
        variety: VarietyConfig
    ) -> str:
        """Detailed analysis of a candidate"""
        
        parts = []
        move = cand.move_uci
        
        # Intro
        intro = random.choice([
            f"Let me look at {move} more carefully.",
            f"What about {move}?",
            f"Consider {move}.",
            f"The move {move} is interesting.",
        ])
        parts.append(intro)
        
        # Idea
        parts.append(self._move_idea_detailed(cand))
        
        # Variation
        pv_length = random.randint(variety.min_pv_length, variety.max_pv_length)
        pv = " ".join(cand.pv_uci[:pv_length])
        parts.append(f"The line might continue: {pv}")
        
        # Patterns
        if cand.patterns:
            parts.append(self._pattern_description(cand.patterns[0], cand))
        
        # Trap warning
        if cand.is_trap and random.random() < variety.include_trap_warning:
            parts.append(self._trap_warning(cand))
        
        # Assessment
        parts.append(self._move_assessment_text(cand))
        
        return " ".join(parts)
    
    def _move_idea(self, cand: CandidateMove) -> str:
        """Brief idea behind a move"""
        if cand.is_check:
            return "gives check, forcing a response"
        if cand.is_capture:
            return f"wins the {cand.captured_piece}" if cand.captured_piece else "makes a capture"
        if "fork" in cand.patterns:
            return "creates a double attack"
        if "pin" in cand.patterns:
            return "creates a pin"
        
        ideas = [
            "improves piece placement",
            "increases control of the center",
            "prepares a future plan",
            "develops a piece",
            "creates threats",
        ]
        return random.choice(ideas)
    
    def _move_idea_detailed(self, cand: CandidateMove) -> str:
        """Detailed explanation of move idea"""
        templates = [
            f"The idea is to {self._move_idea(cand)}.",
            f"This move aims to {self._move_idea(cand)}.",
            f"With this, we're trying to {self._move_idea(cand)}.",
        ]
        return random.choice(templates)
    
    def _move_effect(self, cand: CandidateMove) -> str:
        """What the move accomplishes"""
        if cand.is_check:
            return "gives check"
        if cand.is_capture:
            return f"captures material"
        return "improves the position"
    
    def _quick_assessment(self, cand: CandidateMove) -> str:
        """Quick textual assessment"""
        if cand.win_percent > 70:
            return random.choice(["This looks very strong.", "Clearly good."])
        if cand.win_percent > 55:
            return random.choice(["This seems promising.", "Looks good."])
        if cand.win_percent > 45:
            return random.choice(["About equal.", "Playable."])
        return random.choice(["Not so clear.", "Has some problems."])
    
    def _move_assessment_text(self, cand: CandidateMove) -> str:
        """Text assessment based on WDL"""
        if cand.win_percent > 70:
            return random.choice([
                "This leads to a very favorable position.",
                "White/Black would be clearly better here.",
                "The position after this is quite good.",
            ])
        if cand.win_percent > 55:
            return random.choice([
                "This seems to give a pleasant position.",
                "The resulting position looks comfortable.",
            ])
        if cand.win_percent > 45:
            return random.choice([
                "The position remains balanced.",
                "Neither side has a clear advantage after this.",
            ])
        if cand.win_percent > 30:
            return random.choice([
                "This leaves something to be desired.",
                "The position becomes a bit uncomfortable.",
            ])
        return random.choice([
            "This doesn't look right.",
            "There seem to be problems with this move.",
        ])
    
    def _brief_line(self, cand: CandidateMove) -> str:
        """Brief variation"""
        pv = " ".join(cand.pv_uci[:3])
        return f"{pv}..."
    
    def _pattern_description(self, pattern: str, cand: CandidateMove) -> str:
        """Describe a tactical pattern in words"""
        
        descriptions = {
            "fork": "This creates a fork, attacking multiple pieces at once.",
            "pin": "There's a pin involved here—a piece can't move without exposing something more valuable.",
            "skewer": "This sets up a skewer through a valuable piece.",
            "discovered_attack": "Moving the piece reveals an attack from behind.",
            "back_rank": "The back rank is weak, and this exploits it.",
            "hanging_piece": "This picks up an undefended piece.",
            "trapped_piece": "The piece has nowhere safe to go.",
        }
        
        return descriptions.get(pattern, f"There's a {pattern} motif here.")
    
    def _trap_warning(self, cand: CandidateMove) -> str:
        """Warning about a trap move"""
        
        if cand.trap_type == "tempting_blunder":
            return f"However, this move is tricky! It looks appealing at first glance, but {cand.trap_description or 'deeper analysis reveals problems'}."
        else:
            return f"Interestingly, this move is better than it first appears. {cand.trap_description or 'It holds hidden resources'}."
    
    def _reconsideration(self, cand: CandidateMove, analysis: AnalysisBundle) -> str:
        """Add a 'wait, actually...' moment"""
        
        templates = [
            f"Actually, looking at {cand.move_uci} again, I notice that...",
            f"Wait, I should double-check {cand.move_uci}...",
            f"Hmm, let me reconsider {cand.move_uci}...",
        ]
        
        base = random.choice(templates)
        
        # Add some observation
        observations = [
            "the position is more complex than I first thought.",
            "there's a tactical nuance I almost missed.",
            "the evaluation isn't as clear-cut as it seemed.",
        ]
        
        return base + " " + random.choice(observations)
    
    def _dead_end(self, analysis: AnalysisBundle) -> str:
        """Show a move that was considered but rejected"""
        
        # Find a worse candidate
        worst = max(analysis.candidates, key=lambda c: c.loss_percent)
        
        templates = [
            f"I also briefly looked at {worst.move_uci}, but it doesn't work because {self._why_bad(worst)}.",
            f"The move {worst.move_uci} is tempting, but it fails to {self._what_it_misses()}.",
            f"Unfortunately, {worst.move_uci} runs into problems: {self._problem_description(worst)}.",
        ]
        
        return random.choice(templates)
    
    def _why_bad(self, cand: CandidateMove) -> str:
        """Explain why a move is bad"""
        reasons = [
            "it leaves pieces undefended",
            "the opponent has a strong reply",
            "it loses control of the center",
            "there's a tactical refutation",
        ]
        return random.choice(reasons)
    
    def _what_it_misses(self) -> str:
        """What a move fails to achieve"""
        return random.choice([
            "address the main threat",
            "maintain the initiative",
            "keep the pieces coordinated",
        ])
    
    def _problem_description(self, cand: CandidateMove) -> str:
        """Describe problems with a move"""
        return random.choice([
            "the position becomes too passive",
            "we lose control of key squares",
            "the opponent gets counterplay",
        ])
    
    def _comparison(self, analysis: AnalysisBundle, variety: VarietyConfig) -> str:
        """Compare the candidates"""
        
        # Get best two
        sorted_candidates = sorted(analysis.candidates, key=lambda c: c.win_percent, reverse=True)
        best = sorted_candidates[0]
        second = sorted_candidates[1] if len(sorted_candidates) > 1 else None
        
        if not second:
            return ""
        
        templates = [
            f"Comparing {best.move_uci} and {second.move_uci}: while both have their merits, {best.move_uci} seems more forcing.",
            f"Between {best.move_uci} and {second.move_uci}, the first option leads to a clearer position.",
            f"Looking at {best.move_uci} vs {second.move_uci} — {best.move_uci} addresses the key issues more directly.",
        ]
        
        return random.choice(templates)
    
    def _conclusion(self, analysis: AnalysisBundle, variety: VarietyConfig) -> str:
        """Generate conclusion - THIS IS WHERE WE REVEAL THE BEST MOVE"""
        
        best = analysis.best_move_uci
        pv = " ".join(analysis.best_pv_uci[:variety.max_pv_length])
        
        # Build conclusion
        intro = random.choice([
            "After considering all of this,",
            "Based on this analysis,",
            "Taking everything into account,",
            "Having explored the options,",
            "Given these considerations,",
        ])
        
        move_statement = random.choice([
            f"{best} is the move to play here.",
            f"I would choose {best}.",
            f"{best} stands out as the right choice.",
            f"the move is {best}.",
            f"{best} is what I'd play.",
        ])
        
        # Key reason (pick one of the patterns or features)
        reason = self._key_reason(analysis)
        
        # PV
        pv_intro = random.choice([
            "The main line continues:",
            "Expected play:",
            "The variation goes:",
            "Play should continue:",
        ])
        
        return f"{intro} {move_statement} {reason}\n\n{pv_intro} {pv}"
    
    def _key_reason(self, analysis: AnalysisBundle) -> str:
        """Extract the key reason for the best move"""
        
        best_cand = next((c for c in analysis.candidates if c.move_uci == analysis.best_move_uci), None)
        
        if not best_cand:
            return ""
        
        if best_cand.patterns:
            pattern = best_cand.patterns[0]
            return f"The {pattern} theme is decisive here."
        
        if best_cand.is_check:
            return "The check gains crucial time."
        
        if best_cand.is_capture:
            return "Winning material is the priority."
        
        reasons = [
            "It's the most forcing continuation.",
            "This maintains the initiative.",
            "The position becomes much easier to play.",
            "It addresses all the key concerns.",
            "This keeps the maximum pressure on the opponent.",
        ]
        
        return random.choice(reasons)
```

---

## Example Outputs

### Example 1: Thorough Template

**FEN**: `r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4`

```
This position comes from the Italian Game. White to move.

White has a significant edge here. The queen on h5 is aggressively placed, 
and there's pressure on f7. Black's king is still in the center.

Let me consider the candidate moves.
- c4f7: takes on f7 with check
- d2d3: a developing move
- b1c3: develops the knight
- d2d4: fights for the center

What about d2d3? The idea is to support the center and prepare development. 
The line might continue: d2d3 f8e7 b1c3. This is solid but doesn't press the 
advantage immediately. The position remains better for White, but we might be 
missing something more direct.

The move c4f7 is interesting. This captures the f7 pawn with check, forcing 
the king to move. Let me look at this more carefully. The idea is to win 
material and expose the black king. The line might continue: c4f7 e8e7 f7g6. 
This wins a pawn and leaves Black's king awkwardly placed. There's a tactical 
motif here—the king can't castle anymore.

Consider b1c3. The idea is to develop a piece toward the center. The line 
might continue: b1c3 f8e7 d2d3. This is natural development. The position 
after this is quite good, but perhaps not the most challenging for Black.

Actually, looking at c4f7 again, I notice that the position is more complex 
than I first thought. After the king moves, we have various ways to continue 
the attack.

I also briefly looked at a2a3, but it doesn't work because it doesn't address 
the dynamic nature of the position.

Comparing c4f7 and b1c3: while both have their merits, c4f7 seems more 
forcing. It wins material immediately and keeps Black under pressure.

After considering all of this, c4f7 is the move to play here. Winning 
material while exposing the enemy king is the priority.

The main line continues: c4f7 e8e7 f7g6 d7d6 d2d4
```

### Example 2: Quick Template

**FEN**: `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1`

```
The opening position after e2e4. Black to move. An equal position to start.

The candidates worth considering:
- e7e5: classical response, fighting for the center
- c7c5: the Sicilian, creating asymmetry
- e7e6: the French, solid and strategic

e7e5 — challenges the center directly. This leads to open positions.

c7c5 — the Sicilian approach, aiming for counterplay. This creates an 
unbalanced game.

e7e6 — prepares d7d5, solid but slightly passive initially.

Based on this analysis, I would choose e7e5. It's the most direct way to 
contest the center and leads to rich middlegame positions.

Expected play: e7e5 g1f3 b8c6 f1b5
```

### Example 3: Problem-Focused Template

**FEN**: `r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5`

```
What's the challenge here? White needs to continue development while 
maintaining the tension in the center. The immediate question is whether 
to castle or to play more actively first.

First, let me check e1g1 (castling). This tucks the king away safely. But 
wait—does it address the central tension? The line might go e1g1 d7d6 h2h3. 
Safe, but perhaps slow.

What about c2c3? This supports a future d3d4 push. After c2c3 d7d6 d3d4, 
we get a strong center. This seems to combine safety with activity.

I looked at b1c3, but it allows c5b4 with some annoyance.

Let me reconsider e1g1... Actually, castling first makes sense because it's 
always useful, and we keep options open.

Hmm, but c2c3 prepares our main plan more directly.

Having explored the options, c2c3 is what I'd play. It prepares the central 
advance and keeps the game on our terms.

The variation goes: c2c3 d7d6 d3d4 e5d4 c3d4 c5b4 b1c3
```

### Example 4: Intuition + Verification Template

**FEN**: `r2qkb1r/ppp2ppp/2n2n2/3pp1B1/2B1P1b1/3P1N2/PPP2PPP/RN1QK2R w KQkq - 0 6`

```
My first impression: this position has some tactical tension. The pin on f6 
and the counter-pin on f3 create complications. Instinctively, I want to 
resolve the tension somehow.

The move h2h3 jumps out—challenging the bishop immediately.

Let me verify this intuition. After h2h3, Black has to decide about the 
bishop. If g4h5, then g5f6 and we've traded our pin for theirs. If g4f3, 
we recapture and the tension is resolved in our favor.

Checking the concrete lines: h2h3 g4h5 g5f6 d8f6 and the position is about 
equal but easier to play for White.

Actually, this confirms the instinct. Challenging the bishop is the right 
approach.

Taking everything into account, h2h3 is the move to play here. It forces a 
decision and simplifies favorably.

Play should continue: h2h3 g4h5 g5f6 d8f6 b1c3
```

### Example 5: With Trap Warning

**FEN**: `r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4`

```
This is the Italian Game, Four Knights variation. White to move. The 
position is roughly equal with both sides well-developed.

A few moves deserve attention here.
- d2d3: solid, supporting the center
- d2d4: aggressive, challenging immediately  
- b1c3: natural development
- c4b5: pins the knight

Consider d2d4. The idea is to open the center immediately. The line might 
continue: d2d4 e5d4 e4e5. This looks very active.

However, this move is tricky! It looks appealing at first glance, but after 
e5d4 f3d4 f6e4, Black wins a pawn and has a great position. The tactics 
work in Black's favor here.

Let me look at b1c3 more carefully. The idea is to develop naturally. The 
line might continue: b1c3 f8e7 d2d3. This is solid and keeps all options 
open. The position after this is quite good.

The move d2d3 is interesting. This supports the center calmly. The line 
might continue: d2d3 f8e7 b1c3. Neither side has a clear advantage after 
this, but it's a sensible approach.

After considering all of this, d2d3 stands out as the right choice. It 
avoids the tactical pitfall in d2d4 while maintaining a healthy position.

The main line continues: d2d3 f8e7 e1g1 e8g8 b1c3
```

---

## File Structure

```
chess_reasoning/
├── __init__.py
├── generator.py              # Main ReasoningTraceGenerator class
├── analyzer.py               # Stockfish analysis wrapper
├── eval_breakdown.py         # NNUE eval parsing
├── patterns.py               # Tactical pattern detection
├── openings.py               # ECO/opening lookup
├── tablebases.py             # Syzygy wrapper
├── trap_detector.py          # Shallow vs deep comparison
│
├── synthesis/
│   ├── __init__.py
│   ├── synthesizer.py        # Main trace generation
│   ├── templates.py          # Template definitions
│   ├── phrases.py            # Phrase libraries
│   └── variety.py            # Randomization logic
│
├── data/
│   ├── openings/             # TSV files from lichess-org/chess-openings
│   │   ├── a.tsv
│   │   ├── b.tsv
│   │   ├── c.tsv
│   │   ├── d.tsv
│   │   └── e.tsv
│   └── syzygy/               # Tablebase files (optional)
│
├── tests/
│   ├── test_analyzer.py
│   ├── test_patterns.py
│   ├── test_synthesizer.py
│   └── fixtures/             # Test FENs and expected outputs
│
└── examples/
    ├── basic_usage.py
    ├── batch_processing.py
    └── custom_templates.py
```

---

## Dependencies

### Python Requirements

```
# requirements.txt
python-chess>=1.10.0      # Chess logic + engine communication
```

### External Requirements

1. **Stockfish binary** - Latest version with NNUE
   - Download: https://stockfishchess.org/download/
   - Needed for: All analysis

2. **Opening database** - lichess-org/chess-openings
   - Download: https://github.com/lichess-org/chess-openings
   - Files: a.tsv, b.tsv, c.tsv, d.tsv, e.tsv
   - Needed for: Opening name lookup

3. **Syzygy tablebases** (optional)
   - Download: https://syzygy-tables.info/
   - 3-4-5 man: ~1 GB
   - 6 man: ~150 GB
   - Needed for: Perfect endgame information

---

## Additional Ideas for Human-Like Reasoning

### 1. Opponent Perspective
Sometimes think from opponent's view:
```
"What would I play as Black here? Probably d7d5, challenging the center. 
So I need to be ready for that..."
```

### 2. Plan Statements
Express longer-term ideas:
```
"The plan is to castle, then push f2f4 to attack on the kingside. First, 
let me make sure everything is in place..."
```

### 3. Piece Coordination Commentary
```
"The knight on f3 and bishop on c4 are working together nicely, both 
pointing at f7..."
```

### 4. Tempo Awareness
```
"This move develops with tempo, attacking the queen. That's an extra move 
gained..."
```

### 5. Pattern Recognition Phrases
```
"This reminds me of a typical Sicilian structure..."
"There's a familiar tactical pattern here..."
```

### 6. Calculation Transparency
```
"Let me calculate: after e4e5, if d6e5, then f3e5 attacking the queen...
wait, the queen can go to a5, so that doesn't work. What else?"
```

### 7. Emotional/Aesthetic Comments
```
"This is quite an elegant solution..."
"The position looks a bit messy, but..."
"I don't love this move, but it might be necessary..."
```

### 8. Prophylactic Thinking
```
"Before doing anything, what is Black threatening? The e4 pawn is 
under pressure, so I should address that first..."
```

### 9. Self-Correction
```
"Actually, I think I was wrong about that. Looking again..."
"No wait, that doesn't work because of..."
```

### 10. Confidence Levels
Vary certainty in language:
```
"I'm pretty confident about this move."
"This seems right, though I'm not 100% sure."
"This is definitely the way to go."
"I'm between two moves here..."
```

Implementation note: the generator now supports optional plan and opponent-perspective
lines plus PV pruning (quiet-line summaries) via config flags such as
`include_plan`, `include_opponent_perspective`, and `pv_prune_quiet`.

---

## Summary

This system generates **human-like reasoning traces** for chess positions by:

1. **Collecting rich analysis data** from Stockfish (MultiPV, WDL, eval breakdown)
2. **Detecting patterns** both tactical and positional
3. **Looking up context** (openings, tablebases)
4. **Identifying traps** through depth comparison
5. **Synthesizing natural language** using varied templates
6. **Randomizing exploration order** so best move is discovered, not declared
7. **Concluding with the best move** only at the end, with PV and reasoning

The output reads like a player thinking through a position—exploring, questioning, 
reconsidering, and finally arriving at a conclusion.

**All moves are in UCI notation.**

**Numbers are minimized; text descriptions dominate.**

**The journey matters as much as the destination.**

Additional Ideas for Human-Like Exploration

1. Candidate Elimination Process

Show the "tournament" of moves—explicitly eliminate candidates:

"That leaves us with two real contenders: d2d4 and b1c3. The others don't 
seem to achieve enough..."


2. "What If" Branching

Explore opponent's alternatives:

"But what if Black doesn't take? What if instead of e5d4, they play d7d6? 
Then I could continue with..."


3. Natural Material Assessment

Humans count material in pieces, not centipawns:

"We're up the exchange here (rook for knight)"
"Material is equal, so this comes down to activity"
"Giving up a pawn for this attack seems worth it"


4. Move Ordering Transparency

Show how humans naturally prioritize:

"First let me check all the checks... okay, none of them work. 
What about captures? There's e4d5..."


5. "Stepping Back" Moments

Periodic re-assessment:

"Let me step back for a second. What's actually happening in this position? 
White has more space, but Black's pieces are solid..."


6. Resource Counting

Natural defender/attacker arithmetic:

"The f7 pawn is attacked twice (queen + bishop) and only defended by 
the king. That's a problem..."


7. Endgame Conversion Thinking

Project ahead to simplified positions:

"If we trade queens, the endgame should be winning because of the 
passed a-pawn..."


8. Defensive Double-Checks

Show caution:

"Before I commit to this, is there any trick I'm missing? Let me check 
for any zwischenzugs or surprise defenses..."


9. Simplification as Strategy

Sometimes the best plan is to simplify:

"Actually, maybe I'm overcomplicating this. If I just trade down to an 
endgame, the extra pawn should be enough..."


10. Position Classification

Humans categorize positions:

"This is a typical IQP (isolated queen pawn) position. The themes are 
clear: piece activity vs long-term pawn weakness..."


11. Practical Considerations

Real-world chess thinking:

"This line is objectively best, but it requires very precise play. 
The simpler approach might be more reliable..."


12. Color-Agnostic Roles

Think in terms of attacker/defender:

"The attacking side needs to open lines. The defender should keep 
things closed and trade pieces..."


These additions would make the traces feel like genuine exploration rather than retroactive justification of the engine's choice. The key is that the best move emerges from the process rather than being stated upfront.

---

## Further Enrichment Ideas

### A. Threat and Defense Scan (Fast Pass)
Before listing candidates, add a short "threat scan" to mimic how humans triage:

- Identify immediate checks for both sides
- Identify hanging pieces (undefended and attacked)
- Identify a direct threat (mate, fork, or material win) if present

Example:
```
First, I should check for tactics. The e5 pawn is loose and the king is still in the center, so there might be a forcing line.
```

### B. Tactical Micro-Motifs (Expanded)
Add more tactical patterns for richer language:

- deflection, decoy, clearance, interference
- attraction, overloading, x-ray attack
- mate net, perpetual check, stalemate trap

These can be detected with shallow search + piece-attack queries and used in template phrases.

### C. Positional Signal Bank
Add a compact set of positional cues tied to piece placement and pawn structure:

- good vs bad bishop (pawn color lock)
- pawn majority targets (e.g., queenside majority)
- rook on open file or 7th rank
- outpost squares that cannot be chased by pawns

Use this to enrich assessment lines without numeric scores.

### D. Variation Management and Pruning
Avoid overly long lines by using a "quiescence-like" policy:

1. Expand only checks, captures, and immediate threats
2. If a line is quiet after 2 ply, summarize it in one sentence
3. Prefer the best line length for puzzles, shorter for games

This keeps the reasoning tight and readable.

### E. Consistency Guards (Trace Hygiene)
Add simple guardrails to reduce contradictions:

- If a move is labeled "best", do not call it risky later
- If win probabilities are close, use "unclear" language
- If tablebase says "draw", avoid claiming a win

This preserves trust in the trace.

### F. Source-Conditioned Dialects
Different sources can bias language:

- puzzles: "forcing", "tactical", "must be precise"
- games: "practical", "safe", "keep options open"

This helps the model learn stylistic intent from the source.

### G. Confidence Banding
Convert win probability bands into confidence phrases:

- 0.60+ -> "feels strong"
- 0.50-0.60 -> "seems playable"
- 0.40-0.50 -> "looks risky"

This yields stable, repeatable language without numbers.

### H. Distillation Alignment Note
Keep the trace anchored to Stockfish's best move, but allow the target move to be the game or puzzle move if desired. This matches:

- CE: teaches the reasoning narrative
- KL: teaches the soft move distribution

This separation prevents the reasoning from conflicting with training targets.

### I. Output Guardrails
Add final checks before emitting:

- UCI notation only for moves
- Ensure the best move appears in the conclusion
- Enforce token budget by trimming optional sections
- If something fails, fall back to a short but valid trace

This guarantees training stability.

### J. Two-Phase Narratives
Optional "intuition then verification" flow:

1. Intuition: "The move that jumps out is ..."
2. Verification: "Checking the line ... it holds up."

This reads naturally and fits puzzle-style positions.

## Performance Notes

Reasoning traces can become CPU-heavy if you enable every enrichment knob. In practice:

- Motif detection is the most expensive component (especially anything that iterates legal moves); keep `max_motifs_per_candidate` small and disable `include_motifs` if preprocessing/training throughput becomes the bottleneck.
- Tablebases are only probed for positions with <= 7 pieces; they should not affect midgame throughput.
- Opening lookups are loaded once and then cached; the first run may be slower.
- If training from a preprocessed dataset, prefer reusing the precomputed trace text instead of regenerating it in the collator. Use `reasoning_trace.rerandomize_in_collator: false` (default) for speed, and only enable it if you explicitly want per-batch trace variation.
