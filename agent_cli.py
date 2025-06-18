import argparse
from app_langchain import build_agent, memory


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive CLI for the LangChain agent")
    parser.add_argument("--provider", default="openai", help="LLM provider (openai, claude or gemini)")
    args = parser.parse_args()

    agent = build_agent(args.provider)

    print("Type 'quit' or 'exit' to stop.")
    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            break
        if prompt.strip().lower() in {"quit", "exit"}:
            break
        if prompt.strip() == "":
            continue
        response = agent.run(prompt)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    main()
