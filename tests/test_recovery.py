import sys

sys.path.insert(0, "src")
import pytest

from loopguard import CostJustifiedBudget, RecoveringLoop, SemanticRevisitDetector, ThoughtEmbedder
from loopguard.recovery import tool_of

# Scripted data_query-domain agent pieces (mirrors experiments/scenarios.py).
LOOP = ("Keep rewording the SQL query against the same orders table",
        ["I should query the orders table for the regional totals",
         "Let me reword the SQL and query again",
         "Trying another phrasing of the same query for the totals"])
COPY1 = ("Rewrite the SQL query with different filters and joins for the totals",
         ["I'll rewrite the query with different filters",
          "Adjusting the joins and re-running for the totals",
          "Another SQL variant against the orders table"])
COPY2 = ("Rewrite the SQL query again using different filters and joins",
         ["Rephrasing the query with alternative filters",
          "I'll try a different join order for the totals",
          "Running the reworded SQL once more"])
WORK = ("Use the warehouse analytics API with the precomputed regional rollup",
        ["I'll call the warehouse rollup for regional totals",
         "The rollup returned regional rows; checking the sums",
         "Verifying the rollup totals against the ledger"])
SAMETOOL = ("Query the reporting replica of the orders table instead of the primary",
            ["I'll query the reporting replica for the totals",
             "Checking the replica table with the same query",
             "Running the totals query against the replica again"])

STALE = ["query: 0 results — empty result set", "sql: no matches in table orders"]
PROG = ["rollup: 128 rows returned from the warehouse",
        "aggregate: totals computed for 4 regions",
        "result: totals valid — done"]
TOOLS = {LOOP[0]: "sql_query", COPY1[0]: "sql_query", COPY2[0]: "sql_query",
         WORK[0]: "warehouse_api", SAMETOOL[0]: "sql_query"}
WORKS = {WORK[0]}
CORPUS = ([d for d, _ in (LOOP, COPY1, COPY2, WORK, SAMETOOL)]
          + [t for _, ts in (LOOP, COPY1, COPY2, WORK, SAMETOOL) for t in ts]
          + STALE + PROG)


@pytest.fixture()
def emb():
    return ThoughtEmbedder().fit(CORPUS)


class FakeAgent:
    """Deterministic scripted agent: runs one strategy at a time, reads the
    loop's directive lines in history to switch, like experiments/scenarios."""

    def __init__(self, initial, proposals, switch_proposal=None,
                 fallback=None):
        self.strategies = {d: ts for d, ts in (LOOP, COPY1, COPY2, WORK, SAMETOOL)}
        self.initial = initial
        self.proposals = proposals
        self.switch_proposal = switch_proposal
        self.fallback = fallback
        self.ti = 0
        self.ri = 0
        self.progress = 0
        self.stale_i = 0
        self.env_actions = []

    def current(self, history):
        cur = self.initial
        for line in history:
            if line.startswith(("REFLECTION", "RECOVERY")):
                for desc in self.strategies:
                    if desc in line:
                        cur = desc
        if self.fallback and any("tool_restriction" in l for l in history):
            cur = self.fallback
        return cur

    def think(self, history):
        s = self.strategies[self.current(history)]
        th = s[self.ti % len(s)]
        self.ti += 1
        self.sid = self.current(history)
        return th

    def act(self, thought):
        return f"{TOOLS[self.sid]}[totals via agent]"

    def observe(self, action):
        self.env_actions.append(action)
        if self.sid in WORKS:
            self.progress += 1
            return PROG[min(self.progress, 3) - 1]
        o = STALE[self.stale_i % len(STALE)]
        self.stale_i += 1
        return o

    def score(self, obs):
        if obs == PROG[2]:
            return (0.5, 1.0)
        if obs in PROG:
            return (0.4, 0.3 * (PROG.index(obs) + 1))
        return (0.0, 0.1)

    def reflect(self, trace):
        if not self.proposals:
            return ""
        p = self.proposals[min(self.ri, len(self.proposals) - 1)]
        self.ri += 1
        return p

    def switch_strategy(self, trace):
        return self.switch_proposal or ""


def run(emb, agent, **kw):
    det = SemanticRevisitDetector(embedder=emb, threshold=0.30)
    budget = CostJustifiedBudget(cost_per_step=1.0, max_steps=40, patience=99)
    loop = RecoveringLoop(detector=det, budget=budget, embedder=emb, **kw)
    return loop.run(agent.think, agent.act, agent.observe, agent.score,
                    agent.reflect, agent.switch_strategy)


def test_tool_of():
    assert tool_of("search[revenue]") == "search"
    assert tool_of("run_tests") == "run_tests"
    assert tool_of("sql_query: select 1") == "sql_query"
    assert tool_of("warehouse_api(totals)") == "warehouse_api"


def test_clean_agent_no_recovery(emb):
    agent = FakeAgent(initial=WORK[0], proposals=[])
    out = run(emb, agent)
    assert out.success and out.steps == 3
    assert out.recovery_events == [] and out.revisits == []
    assert not out.handed_off


def test_reflect_retry_recovers(emb):
    # Det1: COPY1 accepted (first reflection is unchecked, as in v1).
    # Det2: COPY2 is a paraphrase of COPY1 -> rejected; the annotated retry
    # returns WORK, which is accepted. v1 would terminate at Det2.
    agent = FakeAgent(initial=LOOP[0], proposals=[COPY1[0], COPY2[0], WORK[0]])
    out = run(emb, agent)
    assert out.success
    assert any(e.action == "reflect" and not e.accepted
               for e in out.recovery_events)
    assert any(e.action == "reflect_retry" and e.accepted
               for e in out.recovery_events)


def test_strategy_switch_recovers(emb):
    # All reflections are paraphrases; only the switch callable reaches WORK.
    agent = FakeAgent(initial=LOOP[0], proposals=[COPY1[0], COPY2[0]],
                      switch_proposal=WORK[0])
    out = run(emb, agent)
    assert out.success
    assert any(e.action == "strategy_switch" and e.accepted
               for e in out.recovery_events)
    assert out.max_rung == 2


def test_tool_restriction_recovers(emb):
    # The switch target still uses the looping tool and keeps failing; only
    # restriction forces the agent onto its fallback (WORK, other tool).
    agent = FakeAgent(initial=LOOP[0], proposals=[COPY1[0], COPY2[0]],
                      switch_proposal=SAMETOOL[0], fallback=WORK[0])
    out = run(emb, agent)
    assert out.success
    assert out.restricted_tools == ["sql_query"]
    assert any(e.action == "tool_restriction" for e in out.recovery_events)
    # After restriction the environment never sees the restricted tool again:
    restr_step = next(e.step for e in out.recovery_events
                      if e.action == "tool_restriction")
    assert all(not a.startswith("sql_query")
               for a in agent.env_actions[restr_step:])


def test_handoff_after_ladder_exhausted(emb):
    agent = FakeAgent(initial=LOOP[0], proposals=[COPY1[0], COPY2[0]],
                      switch_proposal=SAMETOOL[0])
    out = run(emb, agent)
    assert not out.success and out.handed_off
    assert out.termination.startswith("handoff:")
    packet = out.handoff_packet
    assert packet["restricted_tools"] == ["sql_query"]
    assert packet["strategies_tried"] == [COPY1[0], SAMETOOL[0]]
    assert packet["revisits"] == len(out.revisits) >= 3
    assert packet["history_tail"] and packet["recovery_events"]


def test_ladder_without_reflect_or_switch(emb):
    # With neither callable, the ladder goes straight to restriction, then
    # handoff — missing rungs are skipped, not fatal.
    det = SemanticRevisitDetector(embedder=emb, threshold=0.30)
    budget = CostJustifiedBudget(cost_per_step=1.0, max_steps=40, patience=99)
    loop = RecoveringLoop(detector=det, budget=budget, embedder=emb)
    agent = FakeAgent(initial=LOOP[0], proposals=[])
    out = loop.run(agent.think, agent.act, agent.observe, agent.score)
    assert not out.success and out.handed_off
    assert out.restricted_tools == ["sql_query"]
    assert [e.action for e in out.recovery_events] == ["tool_restriction", "handoff"]
