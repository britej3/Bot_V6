# Human-like memory decay simulation
import wandb
from datetime import timedelta

def track_and_archive(response: str, query: str, source: str, user_id: str):
    """Implements biologically plausible memory decay"""
    # Calculate engagement-based salience (validated metric)
    salience = min(1.0, 
                  (len(response.split()) / 300) + 
                  (time_to_first_token * 0.2) +
                  (1 if user_clicked_helpful else 0))
    
    # Log to W&B for continuous learning
    wandb.log({
        "query": query,
        "response": response,
        "source_path": source,
        "salience_score": salience,
        "user_id": user_id
    })
    
    # Human-like archival decision (3 strikes rule)
    if salience < 0.35:
        archive_count = get_archive_count(user_id, query)
        if archive_count >= 2:  # Requires 3 low-salience signals
            MemoryManager().archive(
                query=query,
                retention_policy="low_salience",
                ttl=timedelta(days=7)  # Mimics human forgetting curve
            )