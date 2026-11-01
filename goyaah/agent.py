"""The real estate sales agent: listings, its role prompt, and booked viewings."""

from __future__ import annotations

import json
import threading
import time
import uuid
from importlib import resources
from pathlib import Path


def load_listings() -> dict:
    return json.loads(resources.files("goyaah").joinpath("data/listings.json").read_text())


def _price(value: int) -> str:
    return f"${value:,}"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def build_prompt(listings: dict, booked: set[tuple[str, str]]) -> str:
    """Role prompt in the style PersonaPlex was trained on ("You work for X which is a Y
    and your name is Z. Information: ..."). Booked slots are left out so the agent only
    offers times that are still free.

    Each home's short name ("Cedar Court") is repeated in front of its viewing times. In
    test calls this cut the agent's mix-ups between homes and made it name the home when
    confirming, which the UI needs to offer the booking."""
    parts = []
    for item in listings["listings"]:
        free = [s.replace(" ", " at ", 1) for s in item["slots"] if (item["id"], s) not in booked]
        parts.append(
            f"{item['name']}: {item['address']}, {_count(item['beds'], 'bedroom')}, "
            f"{_count(item['baths'], 'bathroom')}, {_price(item['price'])}, {item['highlights']}. "
            f"{item['name']} viewings: {', '.join(free) if free else 'none left this week'}."
        )
    return (
        f"You work for {listings['agency']} which is a real estate agency and your name is "
        f"{listings['agent_name']}. Information: You help callers find a home and book a viewing. "
        f"{' '.join(parts)} Only offer the viewing times listed for that home."
    )


class Bookings:
    """Viewings booked from the UI, stored as JSON so they survive restarts."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._items: list[dict] = []
        if path.exists():
            self._items = json.loads(path.read_text())

    def all(self) -> list[dict]:
        with self._lock:
            return list(self._items)

    def booked_slots(self) -> set[tuple[str, str]]:
        with self._lock:
            return {(b["listing_id"], b["slot"]) for b in self._items}

    def add(self, listing_id: str, slot: str, name: str) -> dict:
        listings = {item["id"]: item for item in load_listings()["listings"]}
        if listing_id not in listings:
            raise ValueError(f"unknown listing {listing_id!r}")
        if slot not in listings[listing_id]["slots"]:
            raise ValueError(f"{slot!r} is not a viewing time for this listing")
        with self._lock:
            if any(b["listing_id"] == listing_id and b["slot"] == slot for b in self._items):
                raise ValueError("that viewing time is already booked")
            booking = {
                "id": uuid.uuid4().hex[:8],
                "listing_id": listing_id,
                "address": listings[listing_id]["address"],
                "slot": slot,
                "name": name.strip() or "Caller",
                "created": int(time.time()),
            }
            self._items.append(booking)
            self._save()
        return booking

    def remove(self, booking_id: str) -> bool:
        with self._lock:
            before = len(self._items)
            self._items = [b for b in self._items if b["id"] != booking_id]
            if len(self._items) != before:
                self._save()
                return True
        return False

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._items, indent=2))
        tmp.replace(self.path)
