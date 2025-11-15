#!/usr/bin/env python3
"""Convert a PGN file into (FEN, move, value) entries for supervised training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Dict, Optional

import chess
import chess.pgn

RESULT_TO_VALUE: Dict[str, int] = {
    "1-0": 1,
    "0-1": -1,
    "1/2-1/2": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a PGN file and emit JSONL records with the FEN before each move, "
            "the move in UCI, and the game result value."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("data/combined_elite.pgn"),
        help="Path to the source PGN file (default: data/combined_elite.pgn).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/combined_elite.jsonl"),
        help="Destination JSONL file that will be created/overwritten.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional limit useful for testing; process at most this many games.",
    )
    parser.add_argument(
        "--keep-unknown-results",
        dest="skip_unknown_results",
        action="store_false",
        help="Treat missing/undecided results as draws instead of skipping them.",
    )
    parser.set_defaults(skip_unknown_results=True)
    return parser.parse_args()


def result_to_value(result: str, skip_unknown: bool) -> Optional[int]:
    """Return the scalar value for a PGN result header."""
    normalized = result.strip()
    if normalized in RESULT_TO_VALUE:
        return RESULT_TO_VALUE[normalized]
    return None if skip_unknown else 0


def write_record(out_fh: IO[str], fen: str, move_uci: str, value: int) -> None:
    out_fh.write(
        json.dumps({"fen": fen, "move": move_uci, "value": value}, separators=(",", ":"))
    )
    out_fh.write("\n")


def convert_games(
    pgn_path: Path,
    output_path: Path,
    max_games: Optional[int],
    skip_unknown_results: bool,
) -> Dict[str, int]:
    stats = {"games_read": 0, "games_written": 0, "positions": 0, "skipped_games": 0}

    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        pgn_path.open("r", encoding="utf-8", errors="ignore") as pgn_file,
        output_path.open("w", encoding="utf-8") as out_file,
    ):
        while True:
            if max_games is not None and stats["games_read"] >= max_games:
                break

            try:
                game = chess.pgn.read_game(pgn_file)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Error while reading PGN: {exc}", file=sys.stderr)
                break

            if game is None:
                break

            stats["games_read"] += 1
            game_value = result_to_value(
                game.headers.get("Result", "*"), skip_unknown_results
            )
            if game_value is None:
                stats["skipped_games"] += 1
                continue

            board = game.board()
            wrote_positions = False
            for move in game.mainline_moves():
                fen_before = board.fen()
                mover_is_white = board.turn
                board.push(move)
                if game_value == 0:
                    move_value = 0
                elif mover_is_white:
                    move_value = game_value
                else:
                    move_value = -game_value
                write_record(out_file, fen_before, move.uci(), move_value)
                stats["positions"] += 1
                wrote_positions = True

            if wrote_positions:
                stats["games_written"] += 1
            else:
                stats["skipped_games"] += 1

    return stats


def main() -> None:
    args = parse_args()
    try:
        stats = convert_games(
            args.input, args.output, args.max_games, args.skip_unknown_results
        )
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return

    summary = (
        f"Processed {stats['games_read']} games, wrote {stats['positions']} positions "
        f"from {stats['games_written']} games, skipped {stats['skipped_games']}."
    )
    print(summary, file=sys.stderr)


if __name__ == "__main__":
    main()
