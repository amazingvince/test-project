"""
Qwen3 0.6B Model Merge Script: Chess + Reasoning via Task Arithmetic
=====================================================================

OVERVIEW
--------
This script merges a chess-fine-tuned Qwen3 model with the reasoning/thinking
capabilities from the official Qwen3 instruct model. It uses "Task Arithmetic",
a model merging technique that extracts learned capabilities as "task vectors"
and combines them.

YOUR SETUP
----------
- Base Model:      Qwen/Qwen3-0.6B-Base    (pretrained, no instruction tuning)
- Reasoning Model: Qwen/Qwen3-0.6B         (instruct model with thinking mode)
- Chess Model:     Your fine-tune          (trained FROM the instruct model)

THE PROBLEM
-----------
When you fine-tuned the instruct model on chess data, the model likely 
"forgot" some of its reasoning/thinking capabilities. This is called 
catastrophic forgetting. The chess-specific weights overwrote some of 
the weights responsible for chain-of-thought reasoning.

THE SOLUTION: TASK ARITHMETIC
-----------------------------
Task Arithmetic works by:

1. Computing a "task vector" = fine-tuned_model - base_model
   This vector captures WHAT the fine-tuning learned.

2. Adding/combining task vectors to merge capabilities.

For your case:
    reasoning_vector = reasoning_model - base_model
    
This vector contains everything Qwen's team did to add instruction-following
and thinking capabilities to the base model.

We then add this to your chess model:
    merged = chess_model + scale * reasoning_vector

Since your chess model was fine-tuned FROM the instruct model, it already
has partial reasoning capability. Adding the reasoning_vector reinforces
what was lost during chess fine-tuning.

WHY NOT JUST AVERAGE THE MODELS?
--------------------------------
Simple averaging (0.5 * chess + 0.5 * reasoning) dilutes BOTH capabilities.
Task arithmetic is more surgical - it extracts and adds specific capabilities
without diluting what your chess model learned.

Recent research (Nov 2025) confirms Task Arithmetic is the most reliable
merging method for LLMs, outperforming TIES, DARE, and SLERP in systematic
evaluations across Qwen3 and LLaMA models.

SCALE PARAMETER
---------------
The REASONING_SCALE controls how much reasoning to add:

    Scale   Effect
    -----   ------
    0.2     Light touch - minimal reasoning, chess mostly preserved
    0.3     Conservative - safe starting point
    0.5     Balanced - good default for most cases
    0.7     Strong reasoning recovery - may slightly impact chess
    1.0     Full reasoning vector - maximum reasoning, highest risk to chess
    >1.0    Amplified (experimental) - can destabilize the model

Start with 0.5 and adjust based on evaluation:
- If chess accuracy drops too much  -> lower the scale
- If reasoning/thinking is too weak -> raise the scale

QWEN3 THINKING MODE
-------------------
Qwen3 has a unique "thinking mode" that can be toggled:
- Enable:  /think prefix or enable_thinking=True in generation config
- Disable: /no_think prefix or enable_thinking=False

After merging, test both modes to ensure they work correctly.

USAGE
-----
1. Update CHESS_MODEL path below to point to your fine-tuned model
2. Adjust REASONING_SCALE if needed (start with 0.5)
3. Run: python merge_qwen3_chess_reasoning.py
4. Test the output model on both chess and reasoning tasks

REQUIREMENTS
------------
- torch
- transformers
- ~4GB RAM (0.6B model is small enough to merge on CPU)

REFERENCES
----------
- Task Arithmetic: "Editing Models with Task Arithmetic" (Ilharco et al., 2023)
- Qwen3: https://github.com/QwenLM/Qwen3
- Systematic LLM Merging Study: arXiv:2511.21437 (Nov 2025)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from pathlib import Path

# ==============================================================================
# CONFIGURATION - UPDATE THESE
# ==============================================================================

# The original pretrained model (no instruction tuning)
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"

# The instruct model with reasoning/thinking capabilities
REASONING_MODEL = "Qwen/Qwen3-0.6B"

# Your chess fine-tune (fine-tuned FROM Qwen/Qwen3-0.6B)
CHESS_MODEL = "path/to/your/chess-qwen3-0.6b"  # <-- UPDATE THIS

# Where to save the merged model
OUTPUT_PATH = "./qwen3-0.6b-chess-reasoning"

# How much of the reasoning vector to add (see docstring for guidance)
REASONING_SCALE = 0.5

# ==============================================================================
# MERGE IMPLEMENTATION
# ==============================================================================

def load_state_dict(model_path: str) -> dict:
    """
    Load model weights into a dictionary and free the model from memory.
    
    Uses bfloat16 to reduce memory usage. Loads on CPU to avoid GPU OOM
    when loading multiple models.
    """
    print(f"  Loading {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"    [OK] Loaded {len(weights)} parameters")
    return weights


def task_arithmetic_merge(
    base: dict, 
    chess: dict, 
    reasoning: dict, 
    scale: float
) -> dict:
    """
    Perform task arithmetic merge.
    
    Formula:
        reasoning_vector = reasoning - base
        merged = chess + scale * reasoning_vector
    
    This adds the reasoning capabilities (extracted as a task vector) to
    the chess model without diluting the chess-specific learning.
    
    Parameters
    ----------
    base : dict
        Weights from the base pretrained model (Qwen3-0.6B-Base)
    chess : dict  
        Weights from your chess fine-tuned model
    reasoning : dict
        Weights from the instruct/reasoning model (Qwen3-0.6B)
    scale : float
        How much of the reasoning vector to add (0.0 to 1.0+)
    
    Returns
    -------
    dict
        Merged weights
    """
    merged = {}
    merged_count = 0
    skipped_keys = []
    
    for key in chess.keys():
        if key in reasoning and key in base:
            # Check shapes match
            if chess[key].shape == reasoning[key].shape == base[key].shape:
                # Task arithmetic: chess + scale * (reasoning - base)
                reasoning_vector = reasoning[key] - base[key]
                merged[key] = chess[key] + scale * reasoning_vector
                merged_count += 1
            else:
                skipped_keys.append(key)
                merged[key] = chess[key]
        else:
            skipped_keys.append(key)
            merged[key] = chess[key]
    
    print(f"    [OK] Merged {merged_count}/{len(chess)} parameters")
    if skipped_keys:
        print(f"    [WARN] Skipped {len(skipped_keys)} params (shape/key mismatch)")
    
    return merged


def main():
    print("=" * 70)
    print("  Qwen3 0.6B Task Arithmetic Merge: Chess + Reasoning")
    print("=" * 70)
    print(f"""
  Configuration:
    Base (pretrained):     {BASE_MODEL}
    Reasoning (instruct):  {REASONING_MODEL}
    Chess (your tune):     {CHESS_MODEL}
    
    Reasoning scale:       {REASONING_SCALE}
    Output path:           {OUTPUT_PATH}
    
  Formula:
    merged = chess + {REASONING_SCALE} * (reasoning - base)
""")
    print("=" * 70)
    
    # Step 1: Load all model weights
    print("\n[1/4] Loading model weights...")
    base_weights = load_state_dict(BASE_MODEL)
    reasoning_weights = load_state_dict(REASONING_MODEL)
    chess_weights = load_state_dict(CHESS_MODEL)
    
    # Step 2: Perform the merge
    print("\n[2/4] Performing task arithmetic merge...")
    merged_weights = task_arithmetic_merge(
        base=base_weights,
        chess=chess_weights,
        reasoning=reasoning_weights,
        scale=REASONING_SCALE
    )
    
    # Free memory before creating output model
    del base_weights, chess_weights, reasoning_weights
    gc.collect()
    torch.cuda.empty_cache()
    
    # Step 3: Create the merged model
    print("\n[3/4] Creating merged model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )
    model.load_state_dict(merged_weights)
    del merged_weights
    gc.collect()
    
    # Use tokenizer from reasoning model (has chat template + thinking tokens)
    tokenizer = AutoTokenizer.from_pretrained(
        REASONING_MODEL, 
        trust_remote_code=True
    )
    
    # Step 4: Save to disk
    print("\n[4/4] Saving merged model...")
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    # Done!
    print("\n" + "=" * 70)
    print("  [OK] Merge complete!")
    print("=" * 70)
    print(f"""
  Saved to: {OUTPUT_PATH}
  
  Next steps:
    1. Test chess performance:
       - Run your chess evaluation benchmark
       - Compare accuracy to original chess model
       
    2. Test reasoning/thinking:
       - Use /think prefix to enable thinking mode
       - Test on reasoning tasks (math, logic, etc.)
       
    3. Adjust if needed:
       - Chess worse? Lower REASONING_SCALE and re-merge
       - Reasoning weak? Raise REASONING_SCALE and re-merge
       
  Example test code:
  
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        "{OUTPUT_PATH}",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_PATH}")
    
    # Test thinking mode
    messages = [{{"role": "user", "content": "/think What is the best move?"}}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
