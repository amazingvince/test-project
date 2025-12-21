"""
Chess LLM SFT Training - Source Module

Components:
- chess_utils: Board rendering, move parsing, position utilities
- data_processing: Game and puzzle data loading, preprocessing, loss weighting
- formatting: Chat format conversion for SFT training
- stockfish_eval: Stockfish evaluation for all moves (NEW)
- data_processing_with_eval: Data processing with Stockfish analysis (NEW)
- formatting_with_eval: Formatting with move evaluations in thinking (NEW)
"""

from .chess_utils import (
    ChessPosition,
    render_board_utf,
    parse_movetext,
    get_legal_moves_uci,
    get_first_legal_move,
    validate_uci_move,
    extract_uci_from_response,
    position_from_board,
    setup_position_from_fen,
)

from .data_processing import (
    # Position extraction
    game_to_positions,
    puzzle_to_position,
    
    # Streaming
    stream_game_positions,
    stream_puzzle_positions,
    create_streaming_dataset,
    preprocess_and_save,
    
    # Sampling weights
    get_elo_weight,
    
    # Loss weights
    compute_loss_weight,
    compute_loss_weight_linear,
    compute_loss_weight_gaussian,
    compute_loss_weight_step,
)

from .formatting import (
    position_to_messages,
    position_to_text,
    create_formatting_func,
    create_batch_formatting_func,
    add_messages_column,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_RESPONSE_TEMPLATE,
)

# Stockfish evaluation modules
STOCKFISH_EVAL_AVAILABLE = False
try:
    from .stockfish_eval import (
        MoveEvaluation,
        PositionAnalysis,
        StockfishEvaluator,
        analyze_positions_batch,
        compute_move_reward,
        compute_loss_weight_from_cp,
    )
    
    from .data_processing_with_eval import (
        ChessPositionWithEval,
        compute_loss_weight_from_quality,
        position_from_board_with_eval,
        game_to_positions_with_eval,
        puzzle_to_position_with_eval,
        process_positions_batch,
        stream_positions_with_eval,
        preprocess_and_save_with_eval,
    )
    
    from .formatting_with_eval import (
        format_move_analysis,
        DEFAULT_RESPONSE_TEMPLATE_WITH_EVAL,
        COMPACT_RESPONSE_TEMPLATE_WITH_EVAL,
    )
    
    STOCKFISH_EVAL_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Stockfish evaluation modules not available: {e}")

# Policy Distillation modules (new)
DISTILLATION_AVAILABLE = False
try:
    from .stockfish_teacher import (
        StockfishTeacher,
        MoveAnalysis,
        PositionAnalysis as TeacherPositionAnalysis,
        categorize_move,
        cp_to_probability_distribution,
    )
    
    from .formatting_distill import (
        position_to_messages_distill,
        create_distillation_example,
        generate_thinking_text,
        DISTILLATION_PROMPT_TEMPLATE,
        DISTILLATION_RESPONSE_TEMPLATE,
    )

    from .reasoning_trace import ReasoningTraceGenerator
    
    from .distillation_loss import (
        ChessDistillationLoss,
        SequenceDistillationLoss,
        AdaptiveDistillationLoss,
        compute_distillation_metrics,
        create_soft_target_tensor,
    )
    
    from .collator_distill import (
        DistillationCollator,
        PrecomputedDistillationCollator,
    )
    
    DISTILLATION_AVAILABLE = True
except ImportError as e:
    import warnings
    warnings.warn(f"Distillation modules not available: {e}")

__all__ = [
    # Chess utilities
    'ChessPosition',
    'render_board_utf',
    'parse_movetext',
    'get_legal_moves_uci',
    'get_first_legal_move',
    'validate_uci_move',
    'extract_uci_from_response',
    'position_from_board',
    'setup_position_from_fen',
    
    # Data processing - Position extraction
    'game_to_positions',
    'puzzle_to_position',
    
    # Data processing - Streaming
    'stream_game_positions',
    'stream_puzzle_positions',
    'create_streaming_dataset',
    'preprocess_and_save',
    
    # Data processing - Weights
    'get_elo_weight',
    'compute_loss_weight',
    'compute_loss_weight_linear',
    'compute_loss_weight_gaussian',
    'compute_loss_weight_step',
    
    # Formatting
    'position_to_messages',
    'position_to_text',
    'create_formatting_func',
    'create_batch_formatting_func',
    'add_messages_column',
    'DEFAULT_PROMPT_TEMPLATE',
    'DEFAULT_RESPONSE_TEMPLATE',
    
    # Stockfish availability flag
    'STOCKFISH_EVAL_AVAILABLE',
]

# Add stockfish exports if available
if STOCKFISH_EVAL_AVAILABLE:
    __all__.extend([
        # Stockfish evaluation
        'MoveEvaluation',
        'PositionAnalysis',
        'StockfishEvaluator',
        'analyze_positions_batch',
        'compute_move_reward',
        'compute_loss_weight_from_cp',
        
        # Data processing with eval
        'ChessPositionWithEval',
        'compute_loss_weight_from_quality',
        'position_from_board_with_eval',
        'game_to_positions_with_eval',
        'puzzle_to_position_with_eval',
        'process_positions_batch',
        'stream_positions_with_eval',
        'preprocess_and_save_with_eval',
        
        # Formatting with eval
        'format_move_analysis',
        'DEFAULT_RESPONSE_TEMPLATE_WITH_EVAL',
        'COMPACT_RESPONSE_TEMPLATE_WITH_EVAL',
    ])

# Add distillation exports if available
if DISTILLATION_AVAILABLE:
    __all__.extend([
        # Stockfish Teacher
        'StockfishTeacher',
        'MoveAnalysis',
        'categorize_move',
        'cp_to_probability_distribution',
        
        # Distillation formatting
        'position_to_messages_distill',
        'create_distillation_example',
        'generate_thinking_text',
        'DISTILLATION_PROMPT_TEMPLATE',
        'DISTILLATION_RESPONSE_TEMPLATE',
        'ReasoningTraceGenerator',
        
        # Distillation loss
        'ChessDistillationLoss',
        'SequenceDistillationLoss',
        'AdaptiveDistillationLoss',
        'compute_distillation_metrics',
        'create_soft_target_tensor',
        
        # Collators
        'DistillationCollator',
        'PrecomputedDistillationCollator',
        
        # Availability flag
        'DISTILLATION_AVAILABLE',
    ])
