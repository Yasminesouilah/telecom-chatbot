"""
Interactive CLI for the Mobilis Telecom RAG chatbot.

First run will build the knowledge base (embeds every chunk — can
take a minute or two). Subsequent runs reuse the persisted collection
in config.CHROMA_PERSIST_DIR unless you pass --rebuild.
"""

import argparse

from rag.pipeline import chat, setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Re-embed the knowledge base from scratch instead of reusing the persisted collection.",
    )
    args = parser.parse_args()

    print("Setting up knowledge base...")
    setup(rebuild=args.rebuild)
    print("Ready.\n")

    while True:
        message = input("Enter a customer message (or type 'exit' to quit): ").strip()
        if message.lower() == "exit":
            break
        if not message:
            continue

        answer = chat(message)
        print(answer)
        print("-" * 70)


if __name__ == "__main__":
    main()
