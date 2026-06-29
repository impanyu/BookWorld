"""Prompts for the G-Memory mechanism adapted for BookWorld."""


CONDENSE_TRAJECTORY_SYSTEM = (
    "You are an expert at extracting key narrative moments from role-play logs. "
    "Given a round of character interactions, identify the critical events, "
    "decisions, and turning points while filtering out repetitive or trivial exchanges."
)

CONDENSE_TRAJECTORY_USER = """## Event context
{event}

## Full interaction log for this round
{trajectory}

## Instructions
Extract the key moments from this round in chronological order.
- Focus on actions that change the story state, reveal character intent, or create conflict.
- Omit repetitive dialogue, filler, and trivial observations.
- Keep each key moment as a concise one-sentence statement.

Return a numbered list:
1. ...
2. ...
"""


EXTRACT_INSIGHTS_SYSTEM = (
    "You are an advanced reasoning agent that derives general principles from "
    "role-play simulation trajectories. Your insights should be broadly applicable "
    "guidelines about character behavior, narrative dynamics, and collaboration patterns."
)

EXTRACT_INSIGHTS_COMPARE_USER = """You are given two rounds of simulation. The first produced a richer, more engaging narrative. The second was less engaging or had issues.

## Round A (better outcome):
### Event: {event_a}
### Key moments:
{trajectory_a}

## Round B (worse outcome):
### Event: {event_b}
### Key moments:
{trajectory_b}

## Existing insights:
{existing_insights}

By comparing these rounds, update the insight list. Available operations:
- AGREE <N>: <existing insight> (validate an existing insight)
- REMOVE <N>: <existing insight> (contradicted or redundant)
- EDIT <N>: <improved insight> (refine an existing insight)
- ADD: <new insight> (genuinely new principle)

Each insight must follow the "X, because Y" format and be generally applicable.
Do at most 4 operations. Output each on its own line.
"""


EXTRACT_INSIGHTS_SUCCESS_USER = """You are given several rounds of role-play simulation that produced engaging narratives.

## Rounds:
{success_rounds}

## Existing insights:
{existing_insights}

Derive general principles about what made these rounds work well. Available operations:
- AGREE <N>: <existing insight>
- REMOVE <N>: <existing insight>
- EDIT <N>: <improved insight>
- ADD: <new insight>

Each insight must follow the "X, because Y" format.
Do at most 4 operations. Output each on its own line.
"""


MERGE_INSIGHTS_SYSTEM = (
    "You are an agent skilled at consolidating insights. You will receive a list of "
    "insights that may overlap or be redundant. Merge similar ones and output a "
    "refined, concise list. Base your output strictly on the given inputs."
)

MERGE_INSIGHTS_USER = """## Current insights to merge:
{insights}

Consolidate into no more than {limit} refined insights.
Output as a numbered list:
1. ...
2. ...
"""


PROJECT_INSIGHTS_SYSTEM = (
    "You are a context-aware agent. Given a character role and general insights, "
    "adapt the insights into personalized guidance tailored to that character's "
    "personality, goals, and relationships."
)

PROJECT_INSIGHTS_USER = """### Character role:
{role_name}

### Character profile:
{role_profile}

### General insights:
{insights}

### Output (personalized insights for this character):
"""


PROJECT_INSIGHTS_WITH_CONTEXT_USER = """### Current situation:
{context}

### Character role:
{role_name}

### Character profile:
{role_profile}

### General insights:
{insights}

### Output (personalized insights for this character in this situation):
"""


SCORE_RELEVANCE_SYSTEM = "You are an agent that scores the relevance between two narrative contexts on a scale of 1-10."

SCORE_RELEVANCE_USER = """## Past round context:
{past_context}

## Current situation:
{current_context}

How relevant is the past round for informing actions in the current situation?
Score: """
