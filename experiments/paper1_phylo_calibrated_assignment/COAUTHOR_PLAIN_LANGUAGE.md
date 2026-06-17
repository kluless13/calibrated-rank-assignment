# The Work, In Plain Language

*A jargon-free walkthrough for a collaborator coming to this fresh. No statistics
or machine-learning background assumed. Every technical term is explained the
first time it appears, and there's a glossary at the end. For the numbers-dense
version, see [COAUTHOR_BRIEF.md](COAUTHOR_BRIEF.md).*

---

## 1. The problem in one picture

Imagine you scoop a litre of seawater and want to know which fish were recently
nearby. The water carries tiny traces of their DNA — shed skin, scales, waste.
This is **environmental DNA**, or **eDNA**. We read a short, standardized snippet
of that DNA (a **DNA barcode** — think of it as a short ID tag that differs
between species), and we try to name the animal it came from.

The catch: to name a species, you compare its barcode against a **reference
library** of barcodes from already-identified animals. But **most species on
Earth have never been catalogued.** The reference library is full of holes. So
very often the barcode you're holding has **no exact match** — because that
species simply isn't in the library yet.

The standard tools respond by naming the *closest* species in the library
anyway. That's the core mistake we're fixing. Naming the nearest catalogued
species when the real one is missing is **a confident wrong answer** — and in
biodiversity science a confident wrong answer is worse than an honest "I'm not
sure."

## 2. The idea

We treat identification the way a careful doctor treats a diagnosis. A good
doctor who can't pin down the exact illness doesn't invent a precise-sounding
disease — they say the most specific thing the evidence actually supports
("it's a bacterial infection") and stop there, rather than guessing the exact
strain.

Our system does the same with the **tree of life** — the giant family tree that
groups all fish into species, then genera (groups of closely related species),
then families, then orders (progressively broader branches). For each barcode it
returns **the deepest branch it can defend**:

- confident it's a known species? → name the **species**
- not sure of the species but sure of the group? → name the **genus**, or
  **family**, or **order**
- not enough evidence for any of these? → say **"no call"** (an honest
  abstention)

The scientific output is *the most specific claim we can stand behind*, plus a
clear "I don't know" when that's the truth — never a forced species label.

## 3. How the system works (the pipeline)

Think of it as an evidence-gathering assembly line. A barcode goes in; each
station adds a different kind of evidence before a final, careful decision:

1. **Fast lookup.** Pull the most similar barcodes from the library in
   milliseconds (so it scales to huge samples).
2. **Classical checks.** Use long-established sequence-comparison tools (the
   field's trusted workhorses) to measure how close those matches really are.
3. **Family-tree placement + novelty check.** A learned model places the barcode
   onto the tree of life — *even if its species is missing*, it lands near its
   relatives — and flags whether it looks like **something new** not in the
   library at all.
4. **Gap and resolution checks.** Note where the library is thin, and whether
   this particular DNA region is even capable of telling species apart here.
5. **Calibrated decision.** Weigh all of that and return species / genus /
   family / order / no-call, **with the reasons**.
6. **To-do list.** Turn every "no call" into a concrete recommendation: *these
   are the species worth sequencing next to close the gap.*

The design principle: **measure each kind of evidence separately, then combine
them** — rather than trusting a single black-box guess.

## 4. What we found (the headline results, in plain terms)

**It never invents a species.** At our chosen safety setting, across thousands of
held-out test cases, the system made **zero confident wrong species calls** — it
backed off to a broader, correct group instead. Crucially, we proved this holds
even on species the system had *never seen during tuning* (we tuned on one set of
species and tested on a completely separate set — the zero-mistakes result
survived all 30 repeated trials). This was the single most important thing to
get right, and it's solid.

**The model genuinely learned the tree of life.** Its sense of "which barcodes are
close relatives" lines up with the real fish family tree about as well as you
could hope (a correlation of ~0.91, where 1.0 is perfect). We ran two hard checks
to be sure this wasn't an accident:
- We retrained it on a **deliberately scrambled** family tree. Its tree knowledge
  collapsed (0.91 → 0.09) — exactly what should happen if the signal is real.
- We compared against a crude "just count matching DNA letters" baseline (0.38).
  The learned model is more than twice as good.

**It can spot the genuinely new.** When shown barcodes from species outside the
library, it can often tell they're novel rather than forcing them into a known
slot — the ability that makes honest discovery possible.

**We're honest about where we don't win.** For the narrow job of clustering
near-identical barcodes into species groups, the field's classical tools are
still slightly better than our model. We say so plainly. Our model's distinctive
value is elsewhere: understanding the *family tree* structure and *spotting
novelty* — things the classical tools can't do.

## 5. What's genuinely new vs. what we build on

We're scrupulous about credit, because reviewers will be:

- **Building on others (we cite, we don't claim):** placing DNA onto a tree, fast
  library lookup, and grouping sequences into species are all things other
  researchers have done. We use them; we don't claim to have invented them.
- **Genuinely ours:** (a) the **honest "spot the new" ability** inside a
  tree-aware model; (b) **stitching all the evidence into one calibrated
  decision** that returns the deepest defensible rank with a *measured* error
  rate; (c) rigorously **testing under missing references** (deliberately hiding
  species/genera/families to mimic the real, incomplete world); and (d) turning
  abstentions into a **sequencing to-do list**.

## 6. The honest limits

- For water-sample (eDNA) data using a shorter DNA region, **species-level calls
  are often impossible** — the DNA snippet simply doesn't carry enough
  information. We report this as a boundary and deliberately answer at the genus
  or family level there, rather than pretending.
- A second strand of work that bridges between different DNA regions is promising
  but not yet reliable enough at the fine levels; we've set it aside as a
  follow-up paper.
- This is a research tool, not a finished product — the command-line pipeline
  works, but it isn't a polished service yet.

## 7. Where it stands and what's next

The method is **research-complete and stress-tested**: every part works and is
measured, and the two things that could have undermined it (proving the
zero-mistakes result holds on unseen species, and proving the tree knowledge is
real) are both done. The remaining work is **writing the manuscript** and a few
optional refinements.

**Where your input would help most:** Is "give the deepest defensible answer, or
abstain" the right thing to lead with (versus chasing raw species-naming
accuracy)? How central should the water-sample/eDNA strand be in this first
paper? And does the honest positioning against the classical tools read as
credible?

---

## Glossary

- **DNA barcode** — a short, standardized stretch of DNA used as a species ID
  tag; differs between species.
- **Reference library / reference database** — the catalogue of barcodes from
  already-identified organisms that you compare against.
- **environmental DNA (eDNA)** — DNA an organism sheds into its surroundings
  (e.g. into seawater), letting you detect it without seeing it.
- **Tree of life / phylogenetic tree** — the branching family tree of how species
  are related, from species up through genus, family, and order.
- **Species / genus / family / order** — taxonomic levels from most specific
  (species) to broadest (order). A genus is a group of close species; a family a
  group of genera; an order a group of families.
- **Rank** — which taxonomic level an answer is given at.
- **No-call / abstention** — the system declining to answer because the evidence
  is insufficient. A feature, not a failure.
- **Calibration** — tuning the system so that when it commits to an answer, it is
  right a known, controlled fraction of the time.
- **False species call** — confidently naming the wrong species. The error we
  drove to zero.
- **Held-out / unseen** — test data deliberately kept away from the system during
  tuning, so results reflect genuinely new cases.
- **Novelty detection** — recognizing that a barcode comes from something *not* in
  the reference library at all.
- **Correlation** — a number from 0 to 1 measuring how well two things track each
  other; we use it to check how closely the model's sense of relatedness matches
  the real family tree (≈0.91 here).
