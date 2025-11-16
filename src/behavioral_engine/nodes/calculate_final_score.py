
def calculate_final_score(state):
    """
    This function takes the PII score and the MOE score as input and returns a final score based on the aggregation of the
    scores.
    """
    # TODO: Implement final scoring logic
    pii_score = state["pii_score"]
    moe_score = state["moe_score"]
    final_score = pii_score * moe_score
    print(f"--- [Behavior Engine] Final Score: {final_score} ---")
    return {
        "final_score": final_score
    }