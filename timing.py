"""
SongGeneration Studio - Timing History
Track generation times for smart estimates.
"""

import json
from config import TIMING_FILE, MAX_TIMING_RECORDS

# ============================================================================
# Timing History for Smart Estimates
# ============================================================================

def load_timing_history() -> list:
    """Load timing history from file"""
    try:
        if TIMING_FILE.exists():
            with open(TIMING_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[TIMING] Error loading timing history: {e}")
    return []


def save_timing_history(history: list):
    """Save timing history to file"""
    try:
        with open(TIMING_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"[TIMING] Error saving timing history: {e}")


def save_timing_record(metadata: dict):
    """Save a timing record from a completed generation"""
    if metadata.get("generation_time_seconds", 0) <= 0:
        return  # Don't save if no valid timing

    record = {
        "model": metadata.get("model"),
        "num_sections": metadata.get("num_sections", 0),
        "total_lyrics_length": metadata.get("total_lyrics_length", 0),
        "has_lyrics": metadata.get("has_lyrics", False),
        "output_mode": metadata.get("output_mode", "mixed"),
        "has_reference": bool(metadata.get("reference_audio_id")),
        "generation_time_seconds": metadata.get("generation_time_seconds"),
        "completed_at": metadata.get("completed_at"),
    }

    history = load_timing_history()
    history.append(record)

    # Keep only last N records
    if len(history) > MAX_TIMING_RECORDS:
        history = history[-MAX_TIMING_RECORDS:]

    save_timing_history(history)
    print(f"[TIMING] Saved record: {record['model']}, {record['num_sections']} sections, {record['generation_time_seconds']}s")


def get_timing_stats() -> dict:
    """Calculate timing statistics from history for smart estimates"""
    history = load_timing_history()

    if not history:
        return {"has_history": False, "models": {}}

    # Group by model and calculate stats
    model_stats = {}
    for record in history:
        model = record.get("model", "unknown")
        if model not in model_stats:
            model_stats[model] = {
                "times": [],
                "with_lyrics": [],
                "without_lyrics": [],
                "with_reference": [],
                "without_reference": [],
                "by_sections": {},
            }

        time_sec = record.get("generation_time_seconds", 0)
        if time_sec <= 0:
            continue

        model_stats[model]["times"].append(time_sec)

        # Track by lyrics presence
        if record.get("has_lyrics"):
            model_stats[model]["with_lyrics"].append(time_sec)
        else:
            model_stats[model]["without_lyrics"].append(time_sec)

        # Track by reference presence
        if record.get("has_reference"):
            model_stats[model]["with_reference"].append(time_sec)
        else:
            model_stats[model]["without_reference"].append(time_sec)

        # Track by section count
        sections = record.get("num_sections", 0)
        sections_key = str(sections)
        if sections_key not in model_stats[model]["by_sections"]:
            model_stats[model]["by_sections"][sections_key] = []
        model_stats[model]["by_sections"][sections_key].append(time_sec)

    # Calculate averages
    result = {"has_history": True, "models": {}}
    for model, stats in model_stats.items():
        if not stats["times"]:
            continue

        result["models"][model] = {
            "avg_time": int(sum(stats["times"]) / len(stats["times"])),
            "min_time": min(stats["times"]),
            "max_time": max(stats["times"]),
            "count": len(stats["times"]),
            "avg_with_lyrics": int(sum(stats["with_lyrics"]) / len(stats["with_lyrics"])) if stats["with_lyrics"] else None,
            "avg_without_lyrics": int(sum(stats["without_lyrics"]) / len(stats["without_lyrics"])) if stats["without_lyrics"] else None,
            "avg_with_reference": int(sum(stats["with_reference"]) / len(stats["with_reference"])) if stats["with_reference"] else None,
            "avg_without_reference": int(sum(stats["without_reference"]) / len(stats["without_reference"])) if stats["without_reference"] else None,
            "by_sections": {
                k: int(sum(v) / len(v)) for k, v in stats["by_sections"].items() if v
            },
        }

    return result
