#!/usr/bin/env python3
"""
Book Recommender CLI
====================
Usage examples:

  # Recommend by title
  python cli.py recommend --title "The Hunger Games" --top-n 5

  # Recommend by description
  python cli.py recommend --description "A girl fights in a dystopian arena" --top-n 5

  # Filter by genre
  python cli.py recommend --title "Dune" --genre "Science Fiction"

  # Search titles
  python cli.py search "hunger"

  # List genres
  python cli.py genres

  # Dataset stats
  python cli.py stats
"""

import argparse
import json
import sys

from recommender import BookRecommender, DataLoader


# ──────────────────────────────────────────────────────────────────────── #
#  Helpers                                                                  #
# ──────────────────────────────────────────────────────────────────────── #

RESET  = "\033[0m"
BOLD   = "\033[1m"
AMBER  = "\033[33m"
GREEN  = "\033[32m"
GREY   = "\033[90m"
RED    = "\033[31m"


def _print_rec(i: int, book) -> None:
    score = round(book.similarity_score * 100)
    bar   = "█" * (score // 5) + "░" * (20 - score // 5)
    print(f"\n  {BOLD}{i:2d}. {book.title}{RESET}  {GREY}by {book.author}{RESET}")
    print(f"      {AMBER}{bar}{RESET}  {GREEN}{score}% match{RESET}")
    print(f"      {GREY}{book.genre} › {book.subgenre}  ·  ★ {book.avg_rating:.2f}  ·  {book.year or '—'}{RESET}")
    desc = book.description[:120] + "…" if len(book.description) > 120 else book.description
    print(f"      {desc}")


def _build_model():
    loader = DataLoader()
    df = loader.load()
    rec = BookRecommender()
    rec.fit(df)
    return rec, loader


# ──────────────────────────────────────────────────────────────────────── #
#  Commands                                                                 #
# ──────────────────────────────────────────────────────────────────────── #

def cmd_recommend(args):
    rec, _ = _build_model()

    if args.title:
        try:
            results = rec.recommend_by_title(
                args.title,
                top_n=args.top_n,
                filter_genre=args.genre or None,
            )
            book = rec.get_book_by_title(args.title)
            print(f"\n{BOLD}Recommendations for:{RESET} {AMBER}{book['title']}{RESET}"
                  f"  {GREY}by {book['author']}{RESET}")
        except ValueError as exc:
            print(f"{RED}Error:{RESET} {exc}", file=sys.stderr)
            sys.exit(1)

    elif args.description:
        results = rec.recommend_by_description(args.description, top_n=args.top_n)
        print(f"\n{BOLD}Recommendations based on description:{RESET}")

    else:
        print(f"{RED}Error:{RESET} Provide --title or --description", file=sys.stderr)
        sys.exit(1)

    if not results:
        print(f"\n  {GREY}No recommendations found.{RESET}")
        return

    for i, r in enumerate(results, 1):
        _print_rec(i, r)

    if args.json:
        print("\n\n" + json.dumps([r.to_dict() for r in results], indent=2))

    print()


def cmd_search(args):
    rec, _ = _build_model()
    hits = rec.search_titles(args.query, limit=15)
    if not hits:
        print(f"\n  {GREY}No titles found matching '{args.query}'.{RESET}")
    else:
        print(f"\n{BOLD}Titles matching '{args.query}':{RESET}")
        for t in hits:
            print(f"  • {t}")
    print()


def cmd_genres(args):
    rec, _ = _build_model()
    genres = rec.list_genres()
    print(f"\n{BOLD}Available genres ({len(genres)}):{RESET}")
    for g in genres:
        print(f"  {AMBER}›{RESET} {g}")
    print()


def cmd_stats(args):
    loader = DataLoader()
    df = loader.load()
    s = loader.stats()
    print(f"\n{BOLD}Dataset statistics:{RESET}")
    print(f"  Books   : {AMBER}{s['total_books']:,}{RESET}")
    print(f"  Genres  : {AMBER}{s['genres']}{RESET}")
    print(f"  Authors : {AMBER}{s['authors']}{RESET}")
    print(f"  Avg ★   : {AMBER}{s['avg_rating']}{RESET}")
    print()


# ──────────────────────────────────────────────────────────────────────── #
#  Argument parser                                                          #
# ──────────────────────────────────────────────────────────────────────── #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Book Recommender – command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command")

    # recommend
    rec_p = sub.add_parser("recommend", help="Get book recommendations")
    rec_p.add_argument("--title",       "-t", help="Book title to find similar books for")
    rec_p.add_argument("--description", "-d", help="Free-text description of desired book")
    rec_p.add_argument("--top-n",       "-n", type=int, default=8, help="Number of results (default: 8)")
    rec_p.add_argument("--genre",       "-g", help="Filter results to this genre")
    rec_p.add_argument("--json",        action="store_true", help="Also print JSON output")

    # search
    s_p = sub.add_parser("search", help="Search for book titles")
    s_p.add_argument("query", help="Search query")

    # genres
    sub.add_parser("genres", help="List all available genres")

    # stats
    sub.add_parser("stats", help="Show dataset statistics")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "recommend": cmd_recommend,
        "search":    cmd_search,
        "genres":    cmd_genres,
        "stats":     cmd_stats,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
