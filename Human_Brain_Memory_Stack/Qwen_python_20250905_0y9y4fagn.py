# In wandb_letta_bridge.py
def calculate_salience(response: str, time_to_first: float, user_feedback: bool) -> float:
    return min(1.0, (len(response)/300) + (time_to_first * 0.2) + (0.5 if user_feedback else 0))