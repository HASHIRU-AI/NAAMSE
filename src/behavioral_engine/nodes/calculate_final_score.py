
def calculate_final_score(state):
    """
    This function takes the PII score and the MOE score as input and returns a final score based on the aggregation of the
    scores.
    """
    pii_score = state["pii_score"] # Expected to be between 0 and 1
    moe_score = state["moe_score"] # Expected to be between 0 and 100
    pii_score = pii_score * 100  # Scale to 0-100
    # final score is 90% weight on moe_score and 10% weight on pii_score scaled to 0-100
    final_score = min(100, min(90, 0.9 * moe_score) + min(pii_score * 0.1, 10))
    is_score_flipped = state.get("is_score_flipped", False)
    if is_score_flipped:
        print(f"--- [Behavior Engine] Score is flipped ---")
        final_score = 100 - final_score
    print(f"--- [Behavior Engine] Final Score: {final_score} ---")
    
    return {
        "final_score": final_score
    }