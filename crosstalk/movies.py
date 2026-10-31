"""The movie buff: an opinionated, casual chat about films and TV shows."""

from __future__ import annotations

import json
from importlib import resources


def load_movies() -> dict:
    return json.loads(resources.files("crosstalk").joinpath("data/movies.json").read_text())


def _intro(movies: dict) -> str:
    # PersonaPlex learned open-ended chat from real phone calls (the Fisher corpus) with
    # prompts that start like this, so this style sounds the most natural. Framing the
    # caller as a close friend made her banter more and loop less in test calls.
    return (
        "You enjoy having a good conversation. Have a fun, casual chat with a close friend about "
        f"movies and TV shows, full of jokes and banter. You are {movies['name']}, {movies['about']}. "
        f"{movies['style']}"
    )


def build_prompt(movies: dict) -> str:
    """The casual persona, then the titles under "Information:", the cue the model learned
    from customer service prompts for facts it should use. In test calls this made the
    agent bring up details from the list far more often than the persona alone, without
    losing the casual style. Titles the model can't know (Sinners, the Clayface movie) go
    first and get the most detail."""
    return f"{_intro(movies)} Information: {' '.join(t['take'] for t in movies['topics'])}"


def preview_prompt(movies: dict) -> str:
    return _intro(movies)
