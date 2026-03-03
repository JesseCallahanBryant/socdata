# socdata — Multi-Dataset Sociology Research Tool

## Project Overview
Pure Python conversational sociology research tool. Claude guides users through a 6-stage workflow (topic → variables → hypothesis → method → analysis → interpretation) with support for multiple public datasets via a plugin architecture.

## Architecture
- **CLI**: Click entry point → Rich terminal UI → prompt-toolkit REPL
- **AI**: Anthropic Claude API with dynamic system prompts (base + dataset-specific appendix)
- **Data**: Dataset provider plugins returning pandas DataFrames, cached as Parquet
- **Stats**: statsmodels WLS/GLM with robust SEs, scipy for chi-square

## Key Files
- `socdata/cli.py` — Click entry point (`socdata` command)
- `socdata/session.py` — REPL loop, 6-stage workflow, slash commands, structured output parsing
- `socdata/conversation.py` — Claude client, dynamic system prompt, history management
- `socdata/context.py` — ResearchContext dataclass (dataset, dv, ivs, controls, hypothesis, method, years, weights)
- `socdata/display.py` — Rich tables, panels, banner, coefficient tables
- `socdata/setup_check.py` — API key management (~/.socdata/.env)
- `socdata/datasets/base.py` — DatasetProvider ABC, VariableInfo dataclass
- `socdata/datasets/registry.py` — `@register` decorator, `get_provider()`, `list_providers()`
- `socdata/datasets/cache.py` — Parquet caching at ~/.socdata/cache/
- `socdata/datasets/gss.py` — GSS provider (downloads Stata from NORC, reads with pyreadstat)
- `socdata/datasets/anes.py` — ANES provider (manual download required)
- `socdata/datasets/census.py` — Census/ACS provider (Census API)
- `socdata/datasets/wvs.py` — WVS provider (manual download required)
- `socdata/datasets/ipums.py` — IPUMS provider (manual download required)
- `socdata/stats/engine.py` — Analysis dispatcher
- `socdata/stats/ols.py` — Weighted OLS (statsmodels WLS + HC1)
- `socdata/stats/logistic.py` — Weighted binary logistic (statsmodels GLM binomial)
- `socdata/stats/ordinal.py` — Ordinal logistic (statsmodels OrderedModel)
- `socdata/stats/chisq.py` — Weighted chi-square (scipy)
- `socdata/stats/weights.py` — Weight normalization, NA handling
- `socdata/stats/results.py` — AnalysisResult and CoefficientRow dataclasses

## Adding a New Dataset Provider
1. Create `socdata/datasets/newdata.py`
2. Subclass `DatasetProvider` from `base.py`
3. Implement: `download()`, `list_variables()`, `inspect_variables()`
4. Override `system_prompt_appendix()` with dataset-specific Claude guidance
5. Call `register(NewProvider())` at module level
6. Add import to `socdata/datasets/__init__.py`

## Adding a New Statistical Method
1. Create `socdata/stats/newmethod.py`
2. Function signature: `run_newmethod(df, dv, ivs, controls, weight_var, years_used) -> AnalysisResult`
3. Add dispatch case in `socdata/stats/engine.py`
4. Update `_METHOD_RE` in `session.py` and method list in `conversation.py` system prompt

## Build & Run
```bash
pip install -e .
socdata              # launch
socdata --setup      # reset API key
socdata -m claude-sonnet-4-6  # use different model
```

## pyproject.toml
Uses `build-backend = "setuptools.build_meta"` (not the legacy backend).

## Environment
- Config/cache dir: `~/.socdata/`
- API key: `~/.socdata/.env` (ANTHROPIC_API_KEY)
- Dataset cache: `~/.socdata/cache/` (Parquet files)
- Session exports: `~/socdata_exports/`
