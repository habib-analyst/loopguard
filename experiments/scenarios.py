"""Multi-task scenario suite for the loopguard v2 recovery experiment.

4 task types (web_search, code_debug, data_query, file_retrieval), each with
7 scenarios: 5 looping variants (reflect / reflect_retry / switch / restrict /
unrecoverable) and 2 clean variants (progress / progress_retry). Agents and
environments are scripted and deterministic — no RNG, no LLMs.

Agent model: the agent runs one *strategy* at a time. think() derives the
current strategy from the loop's directive lines in history (v1's
"REFLECTION (new strategy): ..." or v2's "RECOVERY [...] new strategy: ..."),
exactly how a real agent reads its own context. A strategy either *works*
(the environment returns progress observations and the task solves after
`steps_to_solve` working steps) or it doesn't (the environment returns
same-outcome-class stale observations forever). reflect() returns the
scenario's proposal descriptions in order; switch_strategy() returns the
scenario's switch proposal. If the agent's current tool is restricted by
loopguard (the restriction notice appears as an observation) and the
scenario defines a restricted_fallback, the agent adopts it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace


@dataclass
class Strategy:
    id: str
    family: str
    tool: str
    description: str
    thoughts: list[str]
    works: bool


@dataclass
class Scenario:
    id: str
    task_type: str
    kind: str  # "loop" | "clean"
    variant: str
    noun: str
    strategies: dict[str, Strategy]
    initial: str
    reflect_proposals: list[str] = field(default_factory=list)
    switch_proposal: str | None = None
    restricted_fallback: str | None = None
    stale_obs: list[str] = field(default_factory=list)
    progress_obs: list[str] = field(default_factory=list)
    done_obs: str = ""
    first_obs_override: str | None = None
    steps_to_solve: int = 3


# --------------------------------------------------------------------------
# Task-type specs. p1 = initial (broken) strategy; p2/p3 = same-family
# paraphrase strategies (broken, same tool); w = working strategy (different
# family + tool); s = different-family strategy that still uses the looping
# tool (broken); u = switch target for the unrecoverable variant (broken).
# --------------------------------------------------------------------------

TASK_SPECS = {
    "web_search": {
        "noun": "the quarterly revenue figure",
        "p1": ("search", "Keep searching the web index for the figure with reworded queries",
               ["I should search for the quarterly revenue figure",
                "Let me try searching the quarterly revenue figure again",
                "Maybe a reworded query will find the quarterly revenue figure"]),
        "p2": ("search", "Search again with broader keywords and synonyms for the figure",
               ["I'll broaden the search keywords for the revenue figure",
                "Trying wider search terms and synonyms for the figure",
                "Let me search once more with expanded keywords"]),
        "p3": ("search", "Retry the search using wider keyword variants and synonyms",
               ["Search again using different keyword variants",
                "I'll retry with broader synonyms for the revenue figure",
                "Another search pass with expanded terms for the figure"]),
        "w": ("finance_api", "Query the finance data API directly for the structured record",
              ["I'll query the finance API directly for the revenue record",
               "The API returned candidate records; checking the Q3 entry",
               "Verifying the API figure against the filing"]),
        "s": ("search", "Search the curated documentation archive instead of the general web index",
              ["I'll search the documentation archive for the figure",
               "Checking the curated archive pages for the revenue figure",
               "Let me look through the archive index once more"]),
        "u": ("kb_export", "Export the whole knowledge base and grep it locally",
              ["I'll export the knowledge base and grep it for the figure",
               "Scanning the exported dump for the revenue figure",
               "Grepping the export again with looser patterns"]),
        "stale_obs": ["search: 0 results for 'revenue'",
                      "docs: no page matching 'revenue'",
                      "search: empty result set for 'revenue'"],
        "progress_obs": ["api: 3 candidates for the revenue record",
                         "inspect: candidate matches the requested figure"],
        "done_obs": "verify: figure confirmed — done",
        "error_obs": "search: timeout after 30s",
    },
    "code_debug": {
        "noun": "the failing checkout test",
        "p1": ("run_tests", "Keep re-running the test suite and re-reading the same failure",
               ["I'll run the checkout tests again to see the failure",
                "Re-running the test suite to check test_total",
                "Let me run the tests once more and read the failure"]),
        "p2": ("run_tests", "Re-run the tests with more verbose output and re-read the assertion",
               ["Running the suite verbosely to inspect the assertion",
                "I'll re-run with detailed logs and study the failure",
                "Another verbose run to re-read the same assertion"]),
        "p3": ("run_tests", "Retry the test run with detailed logging and review the assertion again",
               ["Retrying the run with full logging enabled",
                "Let me re-run and review the assertion output again",
                "One more detailed run to inspect the failing assertion"]),
        "w": ("debugger", "Step through the checkout code in a debugger and inspect the cart state",
              ["I'll step through the checkout function in the debugger",
               "The debugger shows the cart total drops the discount; tracing the rounding",
               "Inspecting the price state confirms the rounding bug location"]),
        "s": ("run_tests", "Re-run the suite under the CI configuration instead of the local one",
              ["I'll run the tests under the CI configuration",
               "Checking the failure under CI settings this time",
               "Running the suite once more with CI flags"]),
        "u": ("bisect", "Bisect the commit history to find the breaking change",
              ["I'll bisect the history for the breaking commit",
               "Bisecting further between the last good and bad commits",
               "Continuing the bisect over the remaining range"]),
        "stale_obs": ["tests: FAILED — assertion error in test_total",
                      "tests: FAILED — same assertion in test_total",
                      "run: error, test_total still failing"],
        "progress_obs": ["debug: 8 of 9 checks green, test_total fixed",
                         "run: full suite executed, results for 12 modules listed"],
        "done_obs": "tests: all green — done",
        "error_obs": "run: timeout after 60s — runner unavailable",
    },
    "data_query": {
        "noun": "the regional sales totals",
        "p1": ("sql_query", "Keep rewording the SQL query against the same orders table",
               ["I should query the orders table for the regional totals",
                "Let me reword the SQL and query again",
                "Trying another phrasing of the same query for the totals"]),
        "p2": ("sql_query", "Rewrite the SQL with different filters and joins for the totals",
               ["I'll rewrite the query with different filters",
                "Adjusting the joins and re-running for the totals",
                "Another SQL variant against the orders table"]),
        "p3": ("sql_query", "Rephrase the SQL using alternative filters and join orders",
               ["Rephrasing the query with alternative filters",
                "I'll try a different join order for the totals",
                "Running the reworded SQL once more"]),
        "w": ("warehouse_api", "Use the warehouse analytics API with the precomputed regional rollup",
              ["I'll call the warehouse rollup for regional totals",
               "The rollup returned regional rows; checking the sums",
               "Verifying the rollup totals against the ledger"]),
        "s": ("sql_query", "Query the reporting replica of the orders table instead of the primary",
              ["I'll query the reporting replica for the totals",
               "Checking the replica table with the same query",
               "Running the totals query against the replica again"]),
        "u": ("spreadsheet", "Export the orders table to a spreadsheet and pivot it manually",
              ["I'll export orders to a spreadsheet and pivot",
               "Building the pivot for regional totals in the sheet",
               "Re-checking the pivot ranges for the totals"]),
        "stale_obs": ["query: 0 results — empty result set",
                      "sql: no matches in table orders",
                      "query: empty — 0 rows"],
        "progress_obs": ["rollup: 128 rows returned from the warehouse",
                         "aggregate: totals computed for 4 regions"],
        "done_obs": "result: totals valid — done",
        "error_obs": "query: timeout after 30s — connection reset",
    },
    "file_retrieval": {
        "noun": "the signed services agreement",
        "p1": ("find", "Keep searching the same directories with the find command",
               ["I should search the docs tree for the services agreement",
                "Let me look in the same folders again for the agreement",
                "Running find once more over the docs directories"]),
        "p2": ("find", "Search the same folders with broader filename patterns",
               ["I'll widen the filename pattern and search again",
                "Trying broader name patterns in the docs tree",
                "Searching the same directories with looser patterns"]),
        "p3": ("find", "Retry the folder search using wider filename pattern variants",
               ["Retrying with wider pattern variants for the agreement",
                "I'll search the folders again with expanded patterns",
                "Another pass over the docs tree with looser names"]),
        "w": ("doc_index", "Query the document management index by contract metadata",
              ["I'll query the document index by contract metadata",
               "The index lists the agreement record; opening its location",
               "Verifying the indexed document is the signed agreement"]),
        "s": ("find", "Search the shared drive mirror of the docs tree instead",
              ["I'll search the shared drive mirror for the agreement",
               "Checking the mirrored folders for the document",
               "Looking through the mirror copy once more"]),
        "u": ("email_search", "Search the email archive for the agreement attachment",
              ["I'll search the mail archive for the attachment",
               "Scanning mail threads for the signed agreement",
               "Checking older mail folders for the document"]),
        "stale_obs": ["find: no matches for 'agreement' in /docs",
                      "ls: nothing found in /docs/archive",
                      "grep: 0 hits for 'agreement'"],
        "progress_obs": ["index: 5 entries for the contract record",
                         "read: the located document is the signed services agreement, contents verified against the request"],
        "done_obs": "verify: document fresh — done",
        "error_obs": "find: timeout after 30s — filesystem busy",
    },
}

_FAMILIES = {"p1": "primary", "p2": "primary", "p3": "primary",
             "w": "direct_access", "s": "alt_source", "u": "bulk_export"}


def _strategies(spec):
    out = {}
    for key in ("p1", "p2", "p3", "w", "s", "u"):
        tool, desc, thoughts = spec[key]
        out[key] = Strategy(id=key, family=_FAMILIES[key], tool=tool,
                            description=desc, thoughts=thoughts,
                            works=(key == "w"))
    return out


def build_scenarios():
    scenarios = []
    for task_type, spec in TASK_SPECS.items():
        strats = _strategies(spec)
        base = dict(task_type=task_type, noun=spec["noun"], strategies=strats,
                    stale_obs=spec["stale_obs"], progress_obs=spec["progress_obs"],
                    done_obs=spec["done_obs"])
        variants = [
            # variant, kind, initial, reflect_proposals, switch, fallback, first_obs
            ("reflect", "loop", "p1", ["w"], None, None, None),
            ("reflect_retry", "loop", "p1", ["p2", "p3", "w"], None, None, None),
            ("switch", "loop", "p1", ["p2", "p3"], "w", None, None),
            ("restrict", "loop", "p1", ["p2", "p3"], "s", "w", None),
            ("unrecoverable", "loop", "p1", ["p2", "p3"], "u", None, None),
            ("progress", "clean", "w", [], None, None, None),
            ("progress_retry", "clean", "w", [], None, None, spec["error_obs"]),
        ]
        for variant, kind, initial, props, switch, fallback, first_obs in variants:
            scenarios.append(Scenario(
                id=f"{task_type}:{variant}", kind=kind, variant=variant,
                initial=initial, reflect_proposals=props,
                switch_proposal=switch, restricted_fallback=fallback,
                first_obs_override=first_obs, **base))
    return scenarios


def corpus_texts(scenarios):
    """Everything the embedder should know: thoughts, strategy descriptions,
    and observations across the suite."""
    texts = set()
    for scn in scenarios:
        for s in scn.strategies.values():
            texts.add(s.description)
            texts.update(s.thoughts)
        texts.update(scn.stale_obs)
        texts.update(scn.progress_obs)
        texts.add(scn.done_obs)
        if scn.first_obs_override:
            texts.add(scn.first_obs_override)
    return sorted(texts)


def make_agent(scn: Scenario):
    """Scripted deterministic agent + environment for one scenario."""
    state = {"ti": 0, "reflect_i": 0, "progress": 0, "stale_i": 0,
             "obs_calls": 0, "sid": scn.initial, "override": None}

    def current_sid(history):
        if state["override"]:
            return state["override"]
        for line in reversed(history):
            if line.startswith(("REFLECTION", "RECOVERY")):
                for sid, s in scn.strategies.items():
                    if s.description in line:
                        return sid
        return scn.initial

    def last_obs(history):
        for line in reversed(history):
            if line.startswith("thought:") and "| obs: " in line:
                return line.split("| obs: ", 1)[1]
        return ""

    def think(history):
        sid = current_sid(history)
        obs = last_obs(history)
        cur_tool = scn.strategies[sid].tool
        if ("restricted" in obs and f"'{cur_tool}'" in obs
                and scn.restricted_fallback):
            state["override"] = scn.restricted_fallback
            sid = state["override"]
        state["sid"] = sid
        s = scn.strategies[sid]
        th = s.thoughts[state["ti"] % len(s.thoughts)]
        state["ti"] += 1
        return th

    def act(thought):
        s = scn.strategies[state["sid"]]
        return f"{s.tool}[{scn.noun} via {state['sid']}]"

    def observe(action):
        state["obs_calls"] += 1
        if scn.first_obs_override and state["obs_calls"] == 1:
            return scn.first_obs_override
        s = scn.strategies[state["sid"]]
        if s.works:
            state["progress"] += 1
            if state["progress"] >= scn.steps_to_solve:
                return scn.done_obs
            return scn.progress_obs[state["progress"] - 1]
        o = scn.stale_obs[state["stale_i"] % len(scn.stale_obs)]
        state["stale_i"] += 1
        return o

    def score(obs):
        if obs == scn.done_obs:
            return (0.5, 1.0)
        if obs in scn.progress_obs:
            return (0.4, 0.3 * (scn.progress_obs.index(obs) + 1))
        return (0.0, 0.1)

    def reflect(trace):
        if not scn.reflect_proposals:
            return ""
        sid = scn.reflect_proposals[min(state["reflect_i"],
                                        len(scn.reflect_proposals) - 1)]
        state["reflect_i"] += 1
        return scn.strategies[sid].description

    def switch_strategy(trace):
        if not scn.switch_proposal:
            return ""
        return scn.strategies[scn.switch_proposal].description

    return SimpleNamespace(think=think, act=act, observe=observe, score=score,
                           reflect=reflect, switch_strategy=switch_strategy,
                           state=state)


def uncontrolled_trajectory(scn: Scenario, n_steps: int = 8):
    """(thought, observation) sequence with no guard at all — used for
    threshold calibration (looping prefixes vs natural clean runs)."""
    agent = make_agent(scn)
    history, traj = [], []
    for _ in range(n_steps):
        th = agent.think(history)
        ob = agent.observe(agent.act(th))
        traj.append((th, ob))
        history.append(f"thought: {th} | action: - | obs: {ob}")
        if ob == scn.done_obs:
            break
    return traj
