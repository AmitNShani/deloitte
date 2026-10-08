"""CLI harness: ask the agent questions from the terminal.

    python -m scripts.ask "Which airports in New England are strong candidates for terminal expansion?"
    python -m scripts.ask --examples      # the four questions from the assignment, one conversation each
    python -m scripts.ask                 # interactive chat (follow-ups keep context)
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

load_dotenv()

from src.agent.agent import AirportAgent
from src.llm.base import (
    AgentEvent,
    MissingCredentialsError,
)

EXAMPLES = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]


def printer(event: AgentEvent) -> None:
    if event.kind == "text":
        print(event.data, end="", flush=True)
    elif event.kind == "tool_call":
        print(f"\n  [tool] {event.data['name']}({json.dumps(event.data['input'])})", file=sys.stderr)
    elif event.kind == "tool_result" and event.data["is_error"]:
        print(f"  [tool error] {event.data['result']}", file=sys.stderr)
    elif event.kind == "notice":
        print(f"\n  [notice] {event.data}", file=sys.stderr)
    elif event.kind == "usage":
        u = event.data
        print(f"  [usage] in={u['input']} cache_read={u['cache_read']} cache_write={u['cache_write']} "
              f"out={u['output']} model={u['model']}", file=sys.stderr)


def main() -> None:
    try:
        AirportAgent()
    except MissingCredentialsError as exc:
        sys.exit(str(exc))
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()

    if args.examples:
        for q in EXAMPLES:
            print(f"\n{'=' * 100}\nQ: {q}\n{'=' * 100}")
            AirportAgent().ask(q, printer)
            print()
        return
    agent = AirportAgent()
    if args.question:
        agent.ask(args.question, printer)
        print()
        return
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            agent.ask(q, printer)
            print()


if __name__ == "__main__":
    main()
