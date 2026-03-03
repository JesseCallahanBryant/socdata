# socdata

A conversational sociology research assistant powered by Claude AI. Guides you through the full research lifecycle — from developing research questions to running survey-weighted analyses and interpreting results — across multiple major public datasets.

```
 ____              ____        _
/ ___|  ___   ___ |  _ \  __ _| |_ __ _
\___ \ / _ \ / __|| | | |/ _` | __/ _` |
 ___) | (_) | (__ | |_| | (_| | || (_| |
|____/ \___/ \___||____/ \__,_|\__\__,_|
```

## Features

- **Multi-dataset support** — GSS, ANES, Census/ACS, World Values Survey, IPUMS
- **Guided 6-stage workflow** — Topic exploration, variable discovery, hypothesis formalization, method selection, analysis execution, interpretation
- **Survey-weighted statistics** — OLS, logistic, ordinal logistic, and chi-square with robust standard errors
- **Conversational AI** — Claude guides you through each stage, suggests variables, discusses measurement validity, and interprets results in plain language
- **Smart caching** — Datasets download once and cache as Parquet for fast subsequent loads

## Installation

```bash
git clone https://github.com/JesseCallahanBryant/socdata.git
cd socdata
pip install -e .
```

For Census API or IPUMS support:
```bash
pip install -e '.[census]'   # Census/ACS
pip install -e '.[ipums]'    # IPUMS microdata
pip install -e '.[all]'      # everything
```

## Setup

You'll need an [Anthropic API key](https://console.anthropic.com). On first run, socdata will prompt you to enter it:

```bash
socdata
```

The key is saved to `~/.socdata/.env`. To reset it:
```bash
socdata --setup
```

## Usage

```bash
socdata                        # start interactive session
socdata -m claude-sonnet-4-6  # use a different model
```

### Example session

```
You › I'm curious whether education predicts confidence in science

  socdata suggests GSS, generates research questions, and walks you
  through variable selection (CONSCI, EDUC, AGE, SEX), hypothesis
  formalization, and method selection (ordinal logistic regression).

You › let's run the analysis

  ┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━┓
  ┃ Term   ┃ Odds Ratio ┃ 95% CI         ┃  Std.Err ┃  p-value ┃ Sig ┃
  ┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━┩
  │ EDUC   │      0.889 │ [0.88, 0.89]   │    0.003 │   <.0001 │ *** │
  │ AGE    │      1.003 │ [1.00, 1.00]   │    0.001 │   <.0001 │ *** │
  │ SEX    │      1.341 │ [1.29, 1.39]   │    0.018 │   <.0001 │ *** │
  └────────┴────────────┴────────────────┴──────────┴──────────┴─────┘

  Claude interprets: "Each additional year of education is associated
  with 11.1% lower odds of being in a less-confident category..."
```

### Slash commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/datasets` | List available datasets |
| `/dataset GSS` | Select a dataset |
| `/search education` | Search variables in current dataset |
| `/stage` | Show current workflow stage |
| `/context` | Display current research design |
| `/reset` | Clear context and start over |
| `/export` | Save session to Markdown |
| `/quit` | Exit socdata |

## Supported Datasets

| Dataset | Source | Access |
|---------|--------|--------|
| **GSS** | General Social Survey (NORC) | Auto-download |
| **ANES** | American National Election Studies | Manual download |
| **Census/ACS** | U.S. Census Bureau | API key required |
| **WVS** | World Values Survey | Manual download |
| **IPUMS** | IPUMS CPS/ACS microdata | API key required |

## Statistical Methods

All methods use survey weights with robust (HC1) standard errors:

- **OLS** — Continuous DVs via weighted least squares
- **Logistic** — Binary DVs via GLM with binomial family, reports odds ratios
- **Ordinal logistic** — Ordered categorical DVs (3-7 levels), reports odds ratios
- **Chi-square** — Weighted test of independence for two categorical variables

## Adding a New Dataset

socdata uses a plugin architecture. To add a dataset:

1. Create `socdata/datasets/newdata.py`
2. Subclass `DatasetProvider` and implement `download()`, `list_variables()`, `inspect_variables()`
3. Call `register(NewProvider())` at module level
4. Add the import to `socdata/datasets/__init__.py`

See `socdata/datasets/gss.py` for a complete example.

## Requirements

- Python 3.10+
- Anthropic API key
- Dependencies: anthropic, click, rich, prompt-toolkit, pandas, statsmodels, scipy, pyreadstat, pyarrow

## License

MIT
