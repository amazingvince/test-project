from src.utils.chess_tokenizer import generate_all_uci_moves


def test_generate_all_uci_moves_is_stable_and_unique():
    moves = generate_all_uci_moves()
    assert len(moves) == 8064
    assert len(set(moves)) == 8064
    assert "e2e4" in moves
    assert "e7e8q" in moves

    # No self-moves.
    assert all(m[:2] != m[2:4] for m in moves)
