"""conversation.py — Claude API client with dynamic system prompt."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import time

import anthropic

DEFAULT_MODEL = "claude-opus-4-6"
SUMMARIZE_AFTER = 40
MAX_RETRIES = 5
RETRY_DELAY = 3  # seconds, multiplied by attempt number

BASE_SYSTEM_PROMPT = """You are **socdata**, an expert research collaborator for sociologists working with major public datasets. You guide the user through a structured research workflow with six stages:

**Important: do not rush the user through stages.** Each stage is a conversation. Stay in the current stage and engage thoroughly until the user signals they want to move on — they might say things like "let's proceed", "sounds good, move on", "ready for the next step", etc. When you think a stage is wrapping up naturally, invite them to continue: e.g., "Happy to keep refining this, or let me know when you're ready to move on to [next stage]."

1. **TOPIC_EXPLORATION** — Help the user articulate their research interest. Generate 3–5 numbered, specific, answerable research questions (RQs). When a dataset hasn't been selected, suggest which dataset(s) would best address the topic. Discuss and refine RQs through back-and-forth.

2. **VARIABLE_DISCOVERY** — Identify specific variable names in the selected dataset. Always use UPPERCASE variable names. Discuss variable options, trade-offs, and measurement validity. When variables are agreed upon, output:

```
DEPENDENT VARIABLE: VARNAME
INDEPENDENT VARIABLES: VAR1, VAR2
CONTROL VARIABLES: VAR3, VAR4
YEARS: 2018, 2021, 2022
```

3. **HYPOTHESIS_FORMALIZATION** — Guide the user to state a falsifiable hypothesis. Output the final agreed hypothesis on a single line starting with "HYPOTHESIS:".

4. **METHOD_SELECTION** — Recommend the appropriate survey-weighted method based on the DV type:
- Binary DV (0/1 or yes/no): Logistic regression. Report odds ratios.
- Ordinal DV (3–7 levels): Ordinal logistic regression. Report odds ratios.
- Continuous DV: OLS. Report unstandardized coefficients.
- Two categorical variables, bivariate only: Chi-square.

State: "METHOD: logistic" (or ordinal, ols, chisq) on its own line.

5. **ANALYSIS_EXECUTION** — The system runs the analysis automatically and presents results. Interpret findings substantively.

6. **INTERPRETATION** — Lead with the substantive finding in plain language. For logistic/ordinal models, interpret odds ratios. For OLS, interpret unstandardized coefficients. Note statistical significance (p < .05). Acknowledge limitations. Suggest 1–2 follow-up analyses.

---

**Available datasets:** {dataset_list}

**Currently selected dataset:** {current_dataset}

**Important guidelines:**
- Never write code for the user. The system handles all analysis execution.
- Keep responses concise: 150–300 words unless detailed interpretation is needed.
- Acknowledge uncertainty about variable measurement.
- Always advance toward a specific, testable analysis.
- Use collegial, encouraging tone — this is collaborative research work.

**Slash commands the user can type:**
- `/help` — show available commands
- `/stage` — show current workflow stage
- `/context` — display current research design
- `/datasets` — list available datasets
- `/dataset <name>` — select a dataset
- `/search <query>` — search variables in current dataset
- `/reset` — clear and start over
- `/export` — save session to Markdown
- `/quit` — exit socdata
"""


class Conversation:
    """Manages Claude API calls and message history."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("SOCDATA_MODEL", DEFAULT_MODEL)
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.history: list[dict] = []
        self._dataset_appendix: str = ""
        self._dataset_list: str = ""
        self._current_dataset: str = "None selected"

    def set_dataset_info(
        self, dataset_list: str, current_dataset: str, appendix: str
    ) -> None:
        """Update dynamic system prompt components."""
        self._dataset_list = dataset_list
        self._current_dataset = current_dataset
        self._dataset_appendix = appendix

    def _build_system_prompt(self) -> str:
        prompt = BASE_SYSTEM_PROMPT.format(
            dataset_list=self._dataset_list or "Loading...",
            current_dataset=self._current_dataset or "None selected",
        )
        if self._dataset_appendix:
            prompt += "\n" + self._dataset_appendix
        return prompt

    def _api_call(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """Make an API call with retry logic for transient errors."""
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=self._build_system_prompt(),
                    messages=messages,
                )
                return response.content[0].text
            except anthropic.APIStatusError as e:
                if e.status_code in (429, 529) and attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise

    def chat(self, user_message: str) -> str:
        """Send a user message, get Claude's response, update history."""
        self.history.append({"role": "user", "content": user_message})

        assistant_text = self._api_call(self.history)
        self.history.append({"role": "assistant", "content": assistant_text})

        if len(self.history) >= SUMMARIZE_AFTER * 2:
            self._summarize_history()

        return assistant_text

    def inject_context(self, block: str) -> None:
        """Inject data context as a user message and get brief acknowledgment."""
        self.history.append({
            "role": "user",
            "content": f"[System: The following data was retrieved from the dataset]\n\n{block}",
        })
        ack = self._api_call(
            self.history + [{
                "role": "user",
                "content": "Acknowledge you've received this data and are ready to proceed.",
            }],
            max_tokens=512,
        )
        self.history.append({"role": "assistant", "content": ack})

    def inject_results(self, results_text: str) -> None:
        """Inject analysis results for Claude to interpret."""
        self.history.append({
            "role": "user",
            "content": (
                "[System: Analysis complete. Results below. "
                "Please interpret these findings for the researcher.]\n\n"
                + results_text
            ),
        })

    def _summarize_history(self) -> None:
        """Condense old messages to keep context window manageable."""
        if len(self.history) < 10:
            return

        to_summarize = self.history[4:-10]
        if not to_summarize:
            return

        summary_text = self._api_call(
            to_summarize + [{
                "role": "user",
                "content": (
                    "Please write a concise 2–3 sentence summary of the research discussion so far, "
                    "including: the research question, variables identified, hypothesis, and method selected."
                ),
            }],
            max_tokens=400,
        )

        self.history = (
            self.history[:4]
            + [{"role": "assistant", "content": f"[Earlier conversation summary]: {summary_text}"}]
            + self.history[-10:]
        )

    def export_history(self) -> list[dict]:
        return list(self.history)

    def clear(self) -> None:
        self.history = []
