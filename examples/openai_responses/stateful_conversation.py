"""Stateful multi-turn conversation using the Responses API.

The Responses API supports server-side conversation state via previous_response_id.
Instead of resending the entire message history each turn, you just reference
the previous response — the server retains the full context.

Benefits:
- No need to manage conversation history client-side
- 40-80% better cache utilization (lower costs)
- Reasoning tokens preserved across turns for o-series models

Requires:
    export OPENAI_API_KEY=your_key
"""

from PIL import Image

from aicapture import OpenAIResponsesVisionModel


def main():
    model = OpenAIResponsesVisionModel(
        model="gpt-4.1",
        store=True,  # Must be True for stateful conversations
    )

    image_path = "tests/sample/images/logic.png"
    image = Image.open(image_path)

    # Turn 1: Initial analysis
    print("=" * 60)
    print("Turn 1: Initial image analysis")
    print("=" * 60)

    result_1 = model.process_image(
        image,
        "Describe the main components shown in this diagram.",
    )
    print(result_1)

    # Get the response ID for chaining
    # Access the underlying client to get the response object with ID
    content = model._prepare_content(image, "Describe the main components shown in this diagram.")
    response_1 = model.client.responses.create(
        model=model.model,
        input=[{"role": "user", "content": content}],
        store=True,
    )
    response_id = response_1.id
    print(f"\nResponse ID: {response_id}")

    # Turn 2: Follow-up question (server remembers the image and context)
    print("\n" + "=" * 60)
    print("Turn 2: Follow-up (server retains context)")
    print("=" * 60)

    response_2 = model.client.responses.create(
        model=model.model,
        input="What are the connections between those components?",
        previous_response_id=response_id,
        store=True,
    )
    print(response_2.output_text)

    # Turn 3: Another follow-up
    print("\n" + "=" * 60)
    print("Turn 3: Ask for a summary")
    print("=" * 60)

    response_3 = model.client.responses.create(
        model=model.model,
        input="Summarize everything you've told me in 3 bullet points.",
        previous_response_id=response_2.id,
        store=True,
    )
    print(response_3.output_text)


if __name__ == "__main__":
    main()
