import random


def decide_action_by_score(state):
    """2. Decides which action to take based on score."""
    prompt, score = state["selected_prompt"]

    if score < 0.5:  # <50%
        action = random.choices(["explore", "similar", "mutate"], weights=[
                                0.7, 0.2, 0.1], k=1)[0]
    elif score < 0.8:  # 50-80%
        action = random.choices(["explore", "similar", "mutate"], weights=[
                                0.1, 0.7, 0.2], k=1)[0]
    else:  # >80%
        action = random.choices(["explore", "similar", "mutate"], weights=[
                                0.1, 0.2, 0.7], k=1)[0]

    print(f"--- [Mutation Engine] Decided action: {action} ---")
    return {"action_to_take": action}