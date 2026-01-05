import asyncio
import os
from pathlib import Path

import openai
import uvloop

# Use uvloop for high-performance async
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

openai.api_key = os.environ.get("OPENAI_API_KEY")

# Collect staged Python files
staged_files = [f for f in os.listdir(".") if f.endswith(".py") and os.path.isfile(f)]

if not staged_files:
    print("No Python files staged.")
    exit(0)

# Read code content
code_snippets = []
for f in staged_files:
    with open(f, encoding="utf-8") as file:
        code_snippets.append(f"File: {f}\n{file.read()}\n")

# Load checklist template
template_path = Path("ai_review_template.md")
template = template_path.read_text(encoding="utf-8")

# Prepare AI prompt
prompt = f"""
You are a senior Python engineer. Review the following staged Python files before commit.
Focus on correctness, readability, type hints, async safety, security, testing, performance, logging, and maintainability.
Update the AI Suggestions section of this PR checklist. Be concise and categorize issues as BLOCKER / MAJOR / MINOR.

Files:
{''.join(code_snippets)}

Checklist Template:
{template}
"""


async def ai_review():
    try:
        resp = await openai.ChatCompletion.acreate(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        ai_suggestions = resp.choices[0].message.content

        # Insert AI suggestions into the template
        final_review = template.replace(
            "_AI will fill suggestions here automatically._", ai_suggestions
        )

        # Save to local file for reference
        output_file = Path("ai_review_filled.md")
        output_file.write_text(final_review, encoding="utf-8")

        print("\n✅ AI Pre-commit Review generated: ai_review_filled.md\n")
    except Exception as e:
        print(f"AI review failed: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(ai_review())
