"""session.py — Main REPL loop and stage management for socdata."""

from __future__ import annotations

import datetime
import re
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import pandas as pd
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.panel import Panel

from socdata.context import ResearchContext
from socdata.conversation import Conversation
from socdata.datasets.base import VariableInfo
from socdata.datasets.registry import get_provider, list_providers
from socdata.display import (
    console,
    print_claude_response,
    print_coefficient_table,
    print_dataset_table,
    print_error,
    print_help,
    print_info,
    print_stage_header,
    print_variable_table,
)
from socdata.stats.engine import run_analysis


class WorkflowStage(Enum):
    TOPIC_EXPLORATION = "topic_exploration"
    VARIABLE_DISCOVERY = "variable_discovery"
    HYPOTHESIS = "hypothesis_formalization"
    METHOD_SELECTION = "method_selection"
    ANALYSIS = "analysis_execution"
    INTERPRETATION = "interpretation"


# Patterns for extracting structured output from Claude responses
_VAR_SPEC_RE = re.compile(
    r"DEPENDENT VARIABLE:\s*([A-Z_][A-Z0-9_]*)"
    r".*?INDEPENDENT VARIABLES:\s*([A-Z_][A-Z0-9_, ]+)"
    r"(?:.*?CONTROL VARIABLES:\s*([A-Z_][A-Z0-9_, ]*))?"
    r"(?:.*?YEARS:\s*([\d, ]+))?",
    re.DOTALL | re.IGNORECASE,
)
_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS:\s*(.+)", re.IGNORECASE)
_METHOD_RE = re.compile(r"METHOD:\s*(logistic|ordinal|ols|chisq)", re.IGNORECASE)
_DATASET_RE = re.compile(r"DATASET:\s*(\w+)", re.IGNORECASE)

# Natural language: user wants to advance
_ADVANCE_RE = re.compile(
    r"\bmove\s+on\b"
    r"|\bnext\s+step\b"
    r"|\bnext\s+stage\b"
    r"|\blet'?s?\s+(go|move|proceed|continue|advance|do\s+it)\b"
    r"|\bproceed\b"
    r"|\bmove\s+(forward|ahead)\b"
    r"|\bi'?m?\s+ready\b"
    r"|\bready\s+to\s+(move|proceed|go|advance)\b"
    r"|\bgo\s+ahead\b"
    r"|\bsounds?\s+good\b"
    r"|\bgood\s+to\s+go\b"
    r"|\bhappy\s+with\s+this\b"
    r"|\b^(next|proceed|continue|advance|ready)$\b",
    re.IGNORECASE | re.MULTILINE,
)

# Natural language: user wants to run the analysis
_RUN_ANALYSIS_RE = re.compile(
    r"\brun\s+(the\s+)?(analysis|regression|model|it)\b"
    r"|\bexecute\s+(the\s+)?(analysis|model|it)\b"
    r"|\brun\s+it\b"
    r"|\bgo\s+for\s+it\b"
    r"|\blet'?s?\s+run\b",
    re.IGNORECASE,
)


class Session:
    """Drives the socdata research workflow."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.conversation = Conversation(model=model)
        self.context = ResearchContext()
        self.stage = WorkflowStage.TOPIC_EXPLORATION
        self._pt_session = PromptSession(history=InMemoryHistory())
        self._last_results: Optional[dict] = None
        self._vars_inspected = False
        self._df: Optional[pd.DataFrame] = None  # loaded dataset
        self._provider = None

        # Initialize conversation with dataset list
        providers = list_providers()
        ds_list = ", ".join(f"{p.name.upper()} ({p.display_name})" for p in providers)
        self.conversation.set_dataset_info(ds_list, "None selected", "")

    def run(self) -> None:
        print_stage_header(self.stage.value)
        self._greet()

        while True:
            try:
                user_input = self._pt_session.prompt("You › ").strip()
            except (KeyboardInterrupt, EOFError):
                self._handle_quit()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self._handle_slash(user_input):
                    break
                continue

            self._process_message(user_input)

    def _greet(self) -> None:
        providers = list_providers()
        ds_names = ", ".join(p.name.upper() for p in providers)
        greeting = (
            f"Hello! I'm socdata, your multi-dataset sociology research assistant. "
            f"I can help you work with several major public datasets: {ds_names}.\n\n"
            f"Tell me about a sociological topic you'd like to investigate — "
            f"for example, 'I'm curious whether education predicts confidence in science' — "
            f"and I'll help you develop a rigorous research question, identify variables, "
            f"formalize a hypothesis, and run a survey-weighted analysis.\n\n"
            f"You can also start by selecting a dataset with `/dataset GSS` or just dive in "
            f"and I'll suggest which dataset fits your question best."
        )
        print_claude_response(greeting, self.stage.value)

    def _process_message(self, user_input: str) -> None:
        import anthropic

        wants_advance = bool(_ADVANCE_RE.search(user_input))
        wants_analysis = bool(_RUN_ANALYSIS_RE.search(user_input))

        if wants_advance or wants_analysis:
            self._try_advance_stage(wants_analysis)

        # If user wants to run analysis and we're ready, skip chat and go straight to analysis
        if (
            wants_analysis
            and self.stage == WorkflowStage.ANALYSIS
            and self.context.is_ready_for_analysis()
            and self._df is not None
        ):
            self._run_analysis()
            return

        try:
            with console.status("[cyan]Thinking...[/]", spinner="dots"):
                response = self.conversation.chat(user_input)
        except anthropic.AuthenticationError:
            print_error("Invalid API key. Run [bold]socdata --setup[/] to update it.")
            return
        except anthropic.BadRequestError as e:
            if "credit balance is too low" in str(e):
                print_error(
                    "Your Anthropic account has no credits.\n"
                    "Add credits at [bold]console.anthropic.com → Plans & Billing[/]"
                )
            else:
                print_error(f"API error: {e}")
            return
        except anthropic.APIError as e:
            print_error(f"Anthropic API error: {e}")
            return

        self._extract_structured_output(response)
        print_claude_response(response, self.stage.value)

        # Auto-select dataset if Claude recommends one
        if not self.context.dataset:
            m = _DATASET_RE.search(response)
            if m:
                self._select_dataset(m.group(1))

        # Variable inspection fires once variables are confirmed
        if (
            self.stage == WorkflowStage.VARIABLE_DISCOVERY
            and self.context.dv
            and self.context.ivs
            and not self._vars_inspected
            and self._df is not None
        ):
            self._run_variable_inspection()
            self._vars_inspected = True

        # Analysis fires when user explicitly asks and context is complete
        if self.stage == WorkflowStage.ANALYSIS and wants_analysis:
            if self.context.is_ready_for_analysis() and self._df is not None:
                self._run_analysis()
            else:
                missing = []
                if not self.context.dataset:
                    missing.append("dataset")
                if not self.context.dv:
                    missing.append("dependent variable")
                if not self.context.ivs:
                    missing.append("independent variables")
                if not self.context.selected_method:
                    missing.append("method")
                print_info("Not ready to run yet — still need: " + ", ".join(missing))

    def _extract_structured_output(self, text: str) -> None:
        """Parse Claude's response for structured specs."""
        m = _VAR_SPEC_RE.search(text)
        if m:
            self.context.dv = m.group(1).strip().upper()
            self.context.ivs = [
                v.strip().upper() for v in m.group(2).split(",") if v.strip()
            ]
            if m.group(3):
                self.context.controls = [
                    v.strip().upper() for v in m.group(3).split(",") if v.strip()
                ]
            if m.group(4):
                year_strs = re.findall(r"\d{4}", m.group(4))
                self.context.years = [int(y) for y in year_strs]

        m = _HYPOTHESIS_RE.search(text)
        if m:
            self.context.hypothesis = m.group(1).strip()

        m = _METHOD_RE.search(text)
        if m:
            self.context.selected_method = m.group(1).strip().lower()

    # Stage advancement
    _STAGE_ORDER = [
        WorkflowStage.TOPIC_EXPLORATION,
        WorkflowStage.VARIABLE_DISCOVERY,
        WorkflowStage.HYPOTHESIS,
        WorkflowStage.METHOD_SELECTION,
        WorkflowStage.ANALYSIS,
        WorkflowStage.INTERPRETATION,
    ]

    def _try_advance_stage(self, wants_analysis: bool = False) -> None:
        current_idx = self._STAGE_ORDER.index(self.stage)
        if current_idx >= len(self._STAGE_ORDER) - 1:
            return

        blocker = self._advancement_blocker()
        if blocker:
            print_info(f"[yellow]Not quite ready to move on — {blocker}[/]")
            return

        next_stage = self._STAGE_ORDER[current_idx + 1]
        self.stage = next_stage
        print_stage_header(self.stage.value)

    def _advancement_blocker(self) -> str:
        if self.stage == WorkflowStage.TOPIC_EXPLORATION:
            if not self.context.dataset:
                return "we haven't selected a dataset yet (use /dataset <name>)"
        elif self.stage == WorkflowStage.VARIABLE_DISCOVERY:
            if not self.context.dv:
                return "we haven't agreed on a dependent variable yet"
            if not self.context.ivs:
                return "we haven't identified independent variables yet"
        elif self.stage == WorkflowStage.HYPOTHESIS:
            if not self.context.hypothesis:
                return "we haven't formalized a hypothesis yet"
        elif self.stage == WorkflowStage.METHOD_SELECTION:
            if not self.context.selected_method:
                return "we haven't settled on a statistical method yet"
        return ""

    # Dataset management
    def _select_dataset(self, name: str) -> bool:
        """Select a dataset provider and load it."""
        provider = get_provider(name)
        if not provider:
            available = ", ".join(p.name.upper() for p in list_providers())
            print_error(f"Unknown dataset '{name}'. Available: {available}")
            return False

        self.context.dataset = provider.name
        self.context.weight_var = provider.default_weight
        self.context.psu_var = provider.default_psu
        self.context.strata_var = provider.default_strata
        self._provider = provider

        # Load the data
        print_info(f"Loading {provider.display_name}...")
        try:
            with console.status(
                f"[cyan]Loading {provider.display_name} (first time may take a minute)...[/]",
                spinner="dots",
            ):
                self._df = provider.download(self.context.years or None)
        except Exception as e:
            print_error(f"Failed to load dataset: {e}")
            self.context.dataset = ""
            self._provider = None
            return False

        n_vars = len(self._df.columns)
        n_rows = len(self._df)
        print_info(
            f"[green]{provider.display_name} loaded: "
            f"{n_rows:,} observations, {n_vars:,} variables[/]"
        )

        # Update conversation with dataset-specific prompt
        providers = list_providers()
        ds_list = ", ".join(f"{p.name.upper()} ({p.display_name})" for p in providers)
        self.conversation.set_dataset_info(
            ds_list,
            f"{provider.display_name} ({provider.name.upper()})",
            provider.system_prompt_appendix(),
        )

        return True

    def _run_variable_inspection(self) -> None:
        if not self._provider or self._df is None:
            return

        all_vars = list(
            {self.context.dv} | set(self.context.ivs) | set(self.context.controls)
        )

        with console.status("[cyan]Inspecting variables...[/]", spinner="dots"):
            results = self._provider.inspect_variables(self._df, all_vars)

        print_variable_table(results)

        missing = [v for v, info in results.items() if not info.found]
        if missing:
            print_error(f"Variable(s) not found: {', '.join(missing)}")
            print_info("Claude will suggest alternatives.")

        # Inject into conversation
        summary_lines = []
        for var, info in results.items():
            if info.found:
                yrs = info.years_available
                yr_range = (
                    f"{yrs[0]}–{yrs[-1]}" if len(yrs) > 1
                    else str(yrs[0]) if yrs
                    else "unknown"
                )
                summary_lines.append(
                    f"- **{var}**: {info.label} | type={info.type} | "
                    f"n={info.n_valid:,} | years={yr_range}"
                )
            else:
                summary_lines.append(f"- **{var}**: NOT FOUND in dataset")

        context_block = "**Variable Inspection Results:**\n" + "\n".join(summary_lines)
        self.conversation.inject_context(context_block)

        # Set DV type for method selection
        dv_info = results.get(self.context.dv)
        if dv_info and dv_info.found:
            self.context.dv_type = dv_info.type

    def _run_analysis(self) -> None:
        if not self.context.is_ready_for_analysis() or self._df is None:
            print_error("Research context incomplete. Need dataset, DV, IVs, and method.")
            return

        print_stage_header(WorkflowStage.ANALYSIS.value)
        print_info(f"Running {self.context.selected_method} analysis...")

        with console.status(
            f"[cyan]Running {self.context.selected_method} analysis...[/]",
            spinner="dots",
        ):
            result = run_analysis(self.context, self._df)

        result_dict = result.to_dict()
        self._last_results = result_dict

        if result.error:
            print_error(f"Analysis failed: {result.error}")
            return

        print_coefficient_table(result_dict)
        self.stage = WorkflowStage.INTERPRETATION
        print_stage_header(self.stage.value)

        # Build results summary for Claude
        coef_lines = []
        for c in result.coefficients:
            if result.method in ("survey_logistic", "survey_ordinal_logistic"):
                val = f"OR={c.odds_ratio:.3f}" if c.odds_ratio else f"b={c.estimate:.3f}"
            else:
                val = f"b={c.estimate:.3f}"
            sig = (
                "***" if c.p_value < 0.001
                else "**" if c.p_value < 0.01
                else "*" if c.p_value < 0.05
                else "ns"
            )
            coef_lines.append(f"  {c.term}: {val}, p={c.p_value:.4f} {sig}")

        results_text = (
            f"Method: {result.method}\n"
            f"n = {result.n}\n"
            f"Years: {result.years_used}\n"
            f"Weight: {result.weight_var}\n\n"
            "Coefficients:\n" + "\n".join(coef_lines)
        )
        if result.r_squared is not None:
            results_text += f"\nR² = {result.r_squared}"

        if result.chi2_statistic is not None:
            results_text = (
                f"Method: {result.method}\n"
                f"n = {result.n}\n"
                f"χ²({result.chi2_df}) = {result.chi2_statistic:.3f}, "
                f"p = {result.chi2_p_value:.4f}\n"
                f"DV: {result.dv}, IV: {result.iv}"
            )

        self.conversation.inject_results(results_text)

        with console.status("[cyan]Interpreting results...[/]", spinner="dots"):
            interpretation = self.conversation.chat(
                "Please provide a substantive interpretation of these findings."
            )

        print_claude_response(interpretation, self.stage.value)

    # Slash commands
    def _handle_slash(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if session should end."""
        parts = cmd.strip().split(maxsplit=1)
        cmd_name = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd_name == "/help":
            print_help()

        elif cmd_name == "/stage":
            label = self.stage.value.replace("_", " ").title()
            print_info(f"Current stage: [bold]{label}[/]")

        elif cmd_name == "/context":
            summary = self.context.summary_text()
            console.print(Panel(summary, title="Research Context", border_style="cyan"))

        elif cmd_name == "/datasets":
            print_dataset_table(list_providers())

        elif cmd_name == "/dataset":
            if not arg:
                print_error("Usage: /dataset <name>  (e.g., /dataset GSS)")
            else:
                self._select_dataset(arg.strip())

        elif cmd_name == "/search":
            if not arg:
                print_error("Usage: /search <query>  (e.g., /search education)")
            elif not self._provider or self._df is None:
                print_error("Select a dataset first with /dataset <name>")
            else:
                results = self._provider.search_variables(self._df, arg.strip())
                if results:
                    from rich.table import Table
                    table = Table(title=f"Variables matching '{arg}'", border_style="dim")
                    table.add_column("Variable", style="bold")
                    table.add_column("Label")
                    for v in results[:20]:
                        table.add_row(v.name, v.label[:60])
                    console.print(table)
                else:
                    print_info(f"No variables found matching '{arg}'")

        elif cmd_name == "/reset":
            self.context = ResearchContext()
            self.conversation.clear()
            self.stage = WorkflowStage.TOPIC_EXPLORATION
            self._last_results = None
            self._vars_inspected = False
            self._df = None
            self._provider = None
            providers = list_providers()
            ds_list = ", ".join(f"{p.name.upper()} ({p.display_name})" for p in providers)
            self.conversation.set_dataset_info(ds_list, "None selected", "")
            print_stage_header(self.stage.value)
            print_info("Research context cleared. Starting fresh.")
            self._greet()

        elif cmd_name == "/export":
            self._export_session()

        elif cmd_name in ("/quit", "/exit", "/q"):
            self._handle_quit()
            return True

        else:
            print_error(f"Unknown command: {cmd}. Type /help for available commands.")

        return False

    def _handle_quit(self) -> None:
        console.print("\n[dim]Goodbye! Good luck with your research.[/]\n")

    def _export_session(self) -> None:
        export_dir = Path.home() / "socdata_exports"
        export_dir.mkdir(exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = export_dir / f"socdata_session_{ts}.md"

        lines = [
            "# socdata Research Session",
            f"*Exported: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}*\n",
        ]

        if self.context.dv:
            lines += [
                "## Research Context",
                self.context.summary_text(),
                "",
            ]

        lines.append("## Conversation\n")
        for msg in self.conversation.export_history():
            role = msg["role"].capitalize()
            content = msg["content"]
            lines.append(f"### {role}\n{content}\n")

        path.write_text("\n".join(lines), encoding="utf-8")
        print_info(f"Session exported to [bold]{path}[/]")
