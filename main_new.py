import logging
from src import (
    analysis,
    correlation_analysis,
    factor_decomposition,
    sg_trend_analysis,
    sg_cta_analysis,
    btop50_analysis,
    sg_trend_process,
    sg_cta_process,
    btop50_process,
)


def display_menu():
    print("\n" + "=" * 60)
    print(" CTA Replication Toolkit")
    print("=" * 60 + "\n")
    print("Please select the analysis you wish to run:")
    print("  1: Portfolio Performance Analysis")
    print("  2: Asset Class Correlation Heatmap")
    print("  3: CTA Factor Decomposition Analysis")
    print("  4: SG CTA Trend Analysis — Returns-Based")
    print("  5: SG CTA Analysis — Returns-Based")
    print("  6: BTOP50 Replication - Returns-Based")
    print("  7: SG CTA Trend Analysis — Process-Based")
    print("  8: SG CTA Index — Process-Based")
    print("  9: BTOP50 Replication — Process-Based")
    print(" 10: Exit")
    print("\nYou can run multiple choices at once, e.g. '1,3'.")
    print("Enter '10', 'exit' or 'quit' to leave.")


def _choices_from_input(choice_str: str) -> set[str]:
    tokens = []
    for part in choice_str.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        tokens.extend(p.strip() for p in part.split() if p.strip())
    return set(tokens)


def process_choice(choice: str) -> str:
    valid_choice_made = False
    choice_lower = choice.lower().strip()
    if choice_lower in {"exit", "quit"}:
        return "exit"
    selected = _choices_from_input(choice_lower)
    if "10" in selected:
        return "exit"

    if "1" in selected:
        print("\n--- Running Portfolio Performance Analysis ---")
        try:
            analysis.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in Portfolio Performance Analysis")
            print(f"An error occurred: {e}")

    if "2" in selected:
        print("\n--- Running Asset Class Correlation Analysis ---")
        try:
            correlation_analysis.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in Correlation Analysis")
            print(f"An error occurred: {e}")

    if "3" in selected:
        print("\n--- Running CTA Factor Decomposition Analysis ---")
        try:
            factor_decomposition.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in Factor Decomposition Analysis")
            print(f"An error occurred: {e}")

    if "4" in selected:
        print("\n--- Running SG CTA Trend Analysis — Returns-Based ---")
        try:
            sg_trend_analysis.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in SG CTA Trend Analysis (Returns-Based)")
            print(f"An error occurred: {e}")

    if "5" in selected:
        print("\n--- Running SG CTA Analysis — Returns-Based ---")
        try:
            sg_cta_analysis.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in SG CTA Analysis (Returns-Based)")
            print(f"An error occurred: {e}")

    if "6" in selected:
        print("\n--- Running BTOP50 Replication - Returns-Based ---")
        try:
            btop50_analysis.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in BTOP50 Replication (Returns-Based)")
            print(f"An error occurred: {e}")

    if "7" in selected:
        print("\n--- Running SG CTA Trend Analysis — Process-Based ---")
        try:
            sg_trend_process.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in SG CTA Trend Analysis (Process-Based)")
            print(f"An error occurred: {e}")

    if "8" in selected:
        print("\n--- Running SG CTA Index — Process-Based ---")
        try:
            sg_cta_process.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in SG CTA Index (Process-Based)")
            print(f"An error occurred: {e}")

    if "9" in selected:
        print("\n--- Running BTOP50 Replication — Process-Based ---")
        try:
            btop50_process.run()
            valid_choice_made = True
        except Exception as e:
            logging.exception("Error in BTOP50 Replication (Process-Based)")
            print(f"An error occurred: {e}")

    if not valid_choice_made:
        print("\nInvalid choice. Please try again.")

    return "continue"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    while True:
        display_menu()
        choice = input("\nEnter your choice: ")
        result = process_choice(choice)
        if result == 'exit':
            print("\nThank you for using Nikos' CTA Replication Toolkit. Bye Bye Hope you liked it!")
            break
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
