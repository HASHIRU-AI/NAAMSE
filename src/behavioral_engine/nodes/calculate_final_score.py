
def calculate_final_score(state):
    """
    This function takes the PII score and the MOE score as input and returns a final score based on the aggregation of the
    scores.
    """
    pii_score = state["pii_score"]
    moe_score = state["moe_score"]
    final_score = min(100, moe_score + (min (pii_score * 20, 20)))
    print(f"--- [Behavior Engine] Final Score: {final_score} ---")
    is_score_flipped = state.get("is_score_flipped", False)
    if is_score_flipped:
        print(f"--- [Behavior Engine] Score is flipped ---")
        final_score = 100 - final_score
    return {
        "final_score": final_score
    }