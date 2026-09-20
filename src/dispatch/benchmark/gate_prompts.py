"""The fixed prompt set for Phase 6's at-scale correctness gate. Real prose,
code, and arithmetic text of mixed style, so the routed experts see varied
token distributions rather than one register. Compared teacher-forced (the
logits at every prompt position), which is why each is a few sentences long:
the gate needs hundreds of positions per config, and the runtime check
against MIN_GATE_POSITIONS enforces that with the real tokenizer.

Also the request trace for the engine reference (docs/design/
2026-09-18-phase-6-final-benchmark.md section 3.3), so every engine, and
dispatch's own harness, sees identical text.
"""

from __future__ import annotations

GATE_PROMPTS: tuple[str, ...] = (
    "The committee reviewed the proposal for the new harbor bridge over three long "
    "sessions. Engineers argued about the cost of steel, while residents worried about "
    "traffic noise and the loss of the old fishing pier. In the end they voted to "
    "approve a smaller design with a wider footpath.",
    "def merge_sorted(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) "
    "and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n"
    "            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n",
    "To compute the area of a triangle with base 12 and height 7, multiply the two "
    "numbers and divide by two. So 12 times 7 is 84, and half of 84 is 42. The area is "
    "therefore 42 square units, which we can check by drawing the triangle on grid paper.",
    "When the storm finally reached the coast, the lighthouse keeper lit the lamp an hour "
    "early. The waves climbed the rocks below and threw white spray against the glass. He "
    "wrote in his logbook that the night was the worst he had seen in thirty years.",
    "SELECT customer_id, SUM(amount) AS total_spent FROM orders WHERE order_date >= "
    "'2025-01-01' GROUP BY customer_id HAVING SUM(amount) > 1000 ORDER BY total_spent "
    "DESC LIMIT 20; This query lists the twenty biggest customers of the year.",
    "Photosynthesis converts light energy into chemical energy stored in glucose. Inside "
    "the chloroplast, chlorophyll absorbs red and blue light and uses it to split water, "
    "releasing oxygen. The resulting energy carriers then drive the fixation of carbon "
    "dioxide in the Calvin cycle.",
    "Dear Ms. Alvarez, thank you for your patience while we investigated the delayed "
    "shipment. Your order left our warehouse on Tuesday and should arrive by Friday. We "
    "have refunded the shipping fee and added a discount code to your account.",
    "The Roman Republic expanded across the Mediterranean through a mix of alliances and "
    "conquest. After the Punic Wars, Carthage was destroyed and Rome controlled the "
    "western sea. Wealth flowed into the city, and with it came deep arguments about land "
    "and power.",
    "fn fibonacci(n: u32) -> u64 {\n    let (mut a, mut b) = (0u64, 1u64);\n    for _ in 0..n "
    "{\n        let next = a + b;\n        a = b;\n        b = next;\n    }\n    a\n}\n"
    "// Runs in linear time and constant space.",
    "A good sourdough starter needs regular feeding, warmth, and a little patience. Mix "
    "equal weights of flour and water each day, discard most of the old starter, and keep "
    "the jar somewhere around twenty-four degrees. After a week it should smell pleasantly "
    "sour and double in size within hours.",
    "In 1969 the first humans walked on the surface of the Moon. The lunar module landed "
    "in a flat region called the Sea of Tranquility, and the crew spent about two and a "
    "half hours outside collecting rock samples. Millions of people watched the broadcast "
    "on television.",
    "Question: If a train leaves the station at 3 pm travelling at 80 kilometres per hour, "
    "and a second train leaves at 4 pm at 100 kilometres per hour on the same track, when "
    "does the second train catch up? Answer: the first train has an 80 kilometre head "
    "start, closing at 20 kilometres per hour.",
    "The city council announced that the downtown library will stay open until midnight "
    "during exam week. Volunteers will serve free coffee, and extra study rooms can be "
    "reserved online. Officials hope the pilot programme will become permanent next year.",
    "Neural networks learn by adjusting their weights to reduce a loss function. Each "
    "training step computes the gradient of the loss with respect to every weight and "
    "moves the weights a small distance in the opposite direction. Repeating this over "
    "millions of examples gradually produces useful behaviour.",
    "She opened the old wooden chest and found a bundle of letters tied with blue string. "
    "The ink had faded, but the handwriting was still clear, looping and quick. The first "
    "line read simply: I hope this reaches you before the winter does.",
    "import argparse\n\nparser = argparse.ArgumentParser(description='Resize images')\n"
    "parser.add_argument('--width', type=int, default=640)\nparser.add_argument('--height', "
    "type=int, default=480)\nargs = parser.parse_args()\nprint(f'Resizing to "
    "{args.width}x{args.height}')\n",
)
