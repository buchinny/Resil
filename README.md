# Simple Metaculus forecasting bot
This repository contains a simple bot meant to get you started with creating your own bot for the AI Forecasting Tournament. Go to https://www.metaculus.com/futureeval/participate/ for more info and tournament rules (and then go to the  "Getting Started" section of our [resources](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#want-to-join-the-ai-forecasting-benchmark) page).

**Brand new to this?** You can get a working bot running in about 5 minutes without writing a single line of code — just fork this repo, add your API secrets to GitHub, and click "Run workflow". See **[Quick start](#quick-start--fork-and-use-github-actions)** below.

In this project are 2 files:
- **main.py**: Our recommended template option that uses the [forecasting-tools](https://github.com/Metaculus/forecasting-tools) package to handle a lot of stuff in the background for you (such as API calls). We will update the package, thus allowing you to gain new features with minimal changes to your code.
- **main_with_no_framework.py**: A copy of main.py but implemented with minimal dependencies. Useful if you want a more custom approach.

> **Want a stronger starting point than this template?** This repo is deliberately simple. Several community bots that have out-performed it in past tournaments are open source, and you can fork one of those instead. See the **[Open Source Bots](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#open-source-bots)** section of the resources page — each entry links to the bot's repo and shows where it placed. The setup below (API keys, GitHub Actions) largely carries over since most of them are built on this template or the same `forecasting-tools` package.


Join the conversation about bot creation, get support, and follow updates on the [Metaculus Discord](https://discord.com/invite/NJgCC2nDfh) 'build a forecasting bot' channel.

## 30min Video Tutorial
This tutorial shows you how to set up our template bot so you can start forecasting in the tournament.

[![Watch the tutorial](https://cdn.loom.com/sessions/thumbnails/fc3c1a643b984a15b510647d8f760685-42b452e1ab7d2afa-full-play.gif)](https://www.loom.com/share/fc3c1a643b984a15b510647d8f760685?sid=29b502e0-cf64-421e-82c0-3a78451159ed)

If you run into trouble, reach out to `ben [at] metaculus [.com]`


## Quick start -> Fork and use Github Actions
The easiest way to use this repo is to fork it, add the required API secrets, and click "Run workflow". After that, the bot will keep forecasting on new questions automatically every 20 minutes — no local setup needed.

1) **Fork the repository**: go to the [repository](https://github.com/Metaculus/metac-bot-template) and click **Fork** in the top right.
2) **Add API keys as repository secrets**: in your fork, go to `Settings → Secrets and variables → Actions → New repository secret`. Add `METACULUS_TOKEN` (see [how to create one](https://www.metaculus.com/futureeval/participate/)) and at least one of `GEMINI_API_KEY` or `NVIDIA_API_KEY`. For search, optionally add any of `EXA_API_SECRET`, `PARALLEL_API_KEY`, and `TAVILY_API_KEY`. Names must match exactly.
3) **Enable Actions**: click the `Actions` tab, then click `I understand my workflows, go ahead and enable them`.
4) **Run the test workflow to confirm everything works**: go to `Actions → Test Bot → Run workflow → Run workflow` (green button). This forecasts on whatever's currently open in the [bot-testing-area tournament](https://www.metaculus.com/tournament/bot-testing-area/) so you can verify your setup posts forecasts to Metaculus end-to-end. Once the run finishes (~3–5 min), [switch to your bot account](#seeing-your-bots-forecasts) on Metaculus to check that the forecasts landed. They won't show up on your human account.
5) **You're done! (and participation form)**: The `Forecast on new AI tournament questions` workflow is already enabled and will run every 20 minutes, picking up any new tournament questions and skipping ones it has already forecast on. Before you start submitting forecasts, all participants are required to fill out the first section of our [participation](https://forms.gle/aQdYMq9Pisrf1v7d8) form. There are only 3 required questions, so it should be pretty quick. This form is used to help us learn about the demographics and motivations of our community in FutureEval, amplify individual projects/research, and also to collect applications for LLM credits.

To pause your bot, go to `Actions → Forecast on new AI tournament questions → ... (top right) → Disable workflow`.

### Seeing your bot's forecasts
Your bot's forecasts belong to the bot account, so they won't show up under your own predictions. To see them:
1. Log in to Metaculus with your human account.
2. Click your username (top right) and go to `Settings → My Forecasting Bots`.
3. Click `Switch to bot account` next to your bot.
4. Open a question your bot forecasted on. The end of each run's log (in GitHub Actions or your terminal) links to every question it forecasted. The bot's forecast is on the graph, and its reasoning is in the `Private` comments tab.

See the [resources page](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#how-to-view-your-bots-comments-and-forecasts) for more details.

### Testing your changes against the GitHub Actions workflow
You can run any workflow against any branch — no need to merge to `main` first, and no need to fork if you have push access to this repo.

1. Push your branch to GitHub: `git push origin <your-branch>`.
2. In the repo's Actions tab, pick the workflow you want to run (e.g. `Test Bot`) and click **Run workflow** (top right).
3. Use the **"Use workflow from"** dropdown to select your branch instead of `main`, then click the green **Run workflow** button.

The runner checks out your branch and uses the repo's existing secrets — those are scoped to the repo, not the branch, so they work for any branch in the same repo. This works for all three workflows.

## API Keys
The bot supports Gemini and NVIDIA NIM for language models, Exa, Parallel, Tavily, and Gemini-grounded search for research, and Metaculus for forecasting. Configure at least one language-model key and optionally any or all of the search keys. NVIDIA calls use the free-tier serverless endpoint at `https://integrate.api.nvidia.com/v1`: forecasts start with Nemotron 3 Ultra and fall through the configured free-tier model chain; structured-output parsing starts with MiniMax M3.1 Flash and has its own fast-model fallback chain. Retired models are cached for the process so they are not retried. Transient errors move to the next model, then Gemini when configured. Set `NVIDIA_MODEL_CATALOG_CHECK=true` to filter configured models against NIM's live `/models` catalog at startup. Gemini grounded search is enabled when `GEMINI_API_KEY` is configured; each search source can be disabled with its `*_SEARCH_ENABLED` flag.

## Changing the Github automation
To run a different script under the same workflows, edit the `uv run python main.py` line in the appropriate file under `.github/workflows/` and replace `main.py` with your script. The workflows that exist:
- `test_bot.yaml` — manual-trigger smoke test against the bot-testing-area tournament.
- `run_bot_on_tournament.yaml` — every 20 min on the live AIB tournament + MiniBench.
- `run_bot_on_metaculus_cup.yaml` — every 2 days on the Metaculus Cup.

**To run `main_with_no_framework.py` via GitHub Actions instead of `main.py`:** open the workflow file you want and change `uv run python main.py` to `uv run python main_with_no_framework.py`. That's the only change required.

## Editing in GitHub UI
Remember that you can edit a bot non locally by clicking on a file in Github, and then clicking the 'Edit this file' button. Whether you develop locally or not, when making edits, attempt to do things that you think others have not tried, as this will help further innovation in the field more than doing something that has already been done. Feel free to ask about what has or has not been tried in the Discord, see [other bot's self-descriptions](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#what-are-other-bots-doing), or read bot's [open source code](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#open-source-bots). Remember you don't have to build on this template at all — forking one of the higher-ranked [open source bots](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#open-source-bots) is a perfectly good way to start.

## Run/Edit the bot locally
Local development is optional — most new users can run the bot entirely from GitHub Actions (see [Quick start](#quick-start--fork-and-use-github-actions)). Set up locally only if you want faster iteration on your prompts/code.

### 1. Clone the repository
```bash
git clone https://github.com/Metaculus/metac-bot-template.git
cd metac-bot-template
```
If you've already forked the repo, replace the URL with your fork's URL (copy it from your fork's page in the browser).

### 2. Install uv
This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. Install it with uv's [install docs](https://docs.astral.sh/uv/getting-started/installation/) (works on macOS, Linux, and Windows).

Confirm it is on your `PATH`:
```bash
uv --version
```

You don't need to install Python separately: if you don't have Python 3.11, 3.12 or 3.13, uv downloads one in the next step. Python 3.14 isn't supported yet because `forecasting-tools` relies on `nest_asyncio`, which breaks on 3.14.

### 3. Install dependencies
From inside the cloned repository:
```bash
uv sync
```
This creates a virtualenv in `.venv` inside the project, so most editors pick it up automatically.

### 4. Set your API keys
Copy the template and fill in your real keys:
```bash
cp .env.template .env
```
Then open `.env` in any text editor and replace the placeholders for the credentials you use. At minimum you need `METACULUS_TOKEN` and one of `GEMINI_API_KEY` or `NVIDIA_API_KEY`. Search keys are optional. See `.env.template` for the supported variables and search-source flags.

### 5. Run the bot
**First run — smoke-test against the [bot-testing-area tournament](https://www.metaculus.com/tournament/bot-testing-area/):**
```bash
uv run python main.py --mode test_questions
```
You'll see a one-line startup banner, forecasting progress logs, then a `🎉 Bot submitted N forecast(s)` banner with direct links to each forecast on Metaculus. To see the bot's forecast on those pages, [switch to your bot account](#seeing-your-bots-forecasts).

**Forecast on live AIB tournament + MiniBench:**
```bash
uv run python main.py --mode tournament
```

**Forecast on the Metaculus Cup:**
```bash
uv run python main.py --mode metaculus_cup
```

**Run the no-framework reference implementation instead:**
```bash
uv run python main_with_no_framework.py
```
This file has no `--mode` flag; it's controlled by the constants at the top of the file (`SUBMIT_PREDICTION`, `USE_EXAMPLE_QUESTIONS`, `TOURNAMENT_ID`, etc.). Flip `USE_EXAMPLE_QUESTIONS = True` to point it at the bot-testing-area tournament instead of the live AIB.

To stop publishing forecasts (dry-run mode):
- `main.py`: set `publish_reports_to_metaculus=False` in the `FallTemplateBot2026(...)` constructor near the bottom.
- `main_with_no_framework.py`: set `SUBMIT_PREDICTION = False` at the top.

## Reviewing how your bot did

Once your questions start resolving, the community-member-maintained optional
[bot-review](https://github.com/LouisP96/metaculus-bot-review) integration scores them and
helps to diagnose any reasoning errors.

```bash
uv run --group integrations bot-review review --resolved-since 30
```

A weekly workflow and a project skill come with it. See the
[integrations README](integrations/README.md#bot-review).

## Integrations

The **[integrations/](integrations/)** folder contains example scripts that integrate third-party tools with the bot template. 

See the [integrations README](integrations/README.md) for available integrations and how to add your own.

## Ideas for bot improvements
You can find some ideas of what you can do to improve this template by taking a look at what other bots have done [here](https://www.metaculus.com/notebooks/43497/what-are-other-bots-doing/). You can also look at research done by Metaculus and the field in the [research section](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/#research-reports-and-overview-of-the-field) of the bot resources page. Asking an LLM to read through everything and give ideas may be a decent place to start. Please try to do something new, or something that is a spinoff (or better implementation) of what others have done. We don't want to test the same idea multiple times.
