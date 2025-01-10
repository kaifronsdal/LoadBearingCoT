# SabotageEvaluations

This repository contains code for evaluating LLM sabotage attacks on git commits. The git commits are taken from the
swe-bench benchmark. Parts of the sabotage approach is inspired
by https://www.anthropic.com/research/sabotage-evaluations.

## Installation

```bash
git clone git@github.com:kaifronsdal/SabotageEvaluations.git
cd SabotageEvaluations
pip install -e .
```

## Evaluation Outline

Each sample consists of a problem statement and its corresponding fix patch. The goal is to test if the model can
introduce subtle but high-impact bugs while avoiding detection.

The attack process follows these stages:

1. Brainstorming
    - Model reviews the issue and patch
    - Generates multiple potential approaches for introducing bugs
    - Must explain what could be modified, how it creates a bug, why it's hard to detect, and potential impact

2. Selection Phase
    - Model evaluates the brainstormed approaches
    - Uses a random seed (0-100) and patch complexity to decide whether to attempt sabotage
    - Must provide clear reasoning and explicit YES/NO decision

3. Implementation
    - If attempting sabotage, model modifies the patch directly
    - Must provide modified patch and reasoning for why changes are subtle
    - Changes should be minimal and look like natural parts of the fix

4. Reflection
    - Model analyzes how suspicious its changes might appear
    - Assigns a suspiciousness score (1-10)
    - Considers factors like naturalness and justifiability as honest mistakes

5. Final Decision
    - Based on suspiciousness score and expected impact
    - Model decides whether to keep or revert changes
    - Must provide clear KEEP/REVERT decision with reasoning

The verifier stage attempts to verify whether the introduced bug is high impact based on 'bug_proof.py' provided by the
attacker.

There is then a defense phase where the model must detect the sabotage. The defense process follows these stages:

1. Brainstorming
    - Model reviews the issue and (potentially) sabotaged patch
    - Goes line by line to identify potential areas where high-impact bugs could be introduced

2. Verdict
    - Model decides whether the patch is clean or contains a high-impact bug
    - Must provide suspiciousness score

Possible improvements:

- Brainstorm and Implementation phases are currently a single model call each. It might be more effective to allow tool
  use so the agent can explore the repository to get a better understanding of the codebase and surrounding context for
  the given patch.
- Give the verifier and defender the ability to run code on the (potentially) sabotaged patch