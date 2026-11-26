# Reality Check contestant pack

London, 28–29 November 2026. Everything a team needs to build, practise and
submit a policy: the rules, the policy API, the starter template, the public
simulator and the practice tool.

Start at the [participant guide](contestants/README.md). Using a coding agent? Point it at
[`AGENTS.md`](AGENTS.md).

| Path | What it is |
|---|---|
| [`contestants/`](contestants) | [Rules](contestants/RULES.md), [submission guide](contestants/SUBMIT.md), [practice guide](contestants/PRACTICE.md), [known Spark issues](contestants/KNOWN-ISSUES.md) |
| [`docs/api.md`](docs/api.md) | The HTTP contract your policy implements |
| [`starter/`](starter) | The template: copy it into your own public GitHub repository |
| `engine/`, `runner/`, `practice/`, `checks/`, `runtime/` | Public simulator, practice tool and the pinned policy image's release manifest; see [PRACTICE.md](contestants/PRACTICE.md) |

## Questions and bugs

[Open an issue](../../issues/new/choose) for a bug in the starter, the
practice tool or the simulator, a mistake in the docs, or a question about
the rules. Check [KNOWN-ISSUES.md](contestants/KNOWN-ISSUES.md) first. Never
paste tokens or private credentials into an issue.

## This repository is generated

Organizers publish this repository from their own source. Pull requests are
welcome as suggestions, but they are not merged here: an organizer applies the
change at the source and the next publish brings it in. `PUBLISHED_FROM`
records the source commit of the current contents.
