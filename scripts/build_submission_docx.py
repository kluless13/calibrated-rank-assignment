import re, sys, pathlib

figdir = sys.argv[1]
out = sys.argv[2]
src = pathlib.Path("experiments/paper1_phylo_calibrated_assignment/MANUSCRIPT.md").read_text()

# strip the internal working-draft note
src = re.sub(r"\*\*Working draft\.\*\*.*?SOURCE_TABLES\.md\)\.\n\n", "", src, flags=re.S)

# clean the author block (handle possible linter reflow via regex)
src = re.sub(
    r"\*\*Author:\*\* Angad Maniyambath.*?\*\*Corresponding author:\*\*[^\n]*\n",
    "Angad Maniyambath\n\nIndependent Researcher\n\nORCID 0009-0000-0985-4721\n",
    src, flags=re.S)

# remove the end-of-paper Figures manifest (figures go inline instead)
s = src.index("## Figures")
e = src.index("## Declarations")
src = src[:s] + src[e:]

figs = [
 ("fig_pipeline_architecture", "The evidence-compiler pipeline architecture."),
 ("fig4_prospective_calibration", "Species-disjoint prospective calibration: the 0% false-species-call rate survives all 30 repeats."),
 ("fig1_place_audit_controls", "Tree recovery vs. the raw k-mer baseline and the shuffled-tree negative control."),
 ("fig_detect_novelty", "Open-set novelty detection (DETECT): AUROC by rank of novelty."),
 ("fig2_rediscovery_headtohead", "Unsupervised species rediscovery: classical vs. neural (species/genus/family AMI)."),
 ("fig_rediscovery_granularity", "Species AMI vs. cluster granularity: the embedding ties VSEARCH at matched ~1.2k-cluster granularity."),
 ("fig3_tree_species_frontier", "The tree-vs-species Pareto frontier."),
 ("fig_missing_reference_collapse", "Rank collapse under hidden references: the hidden rank drops to zero while broader ranks persist."),
]
for i, (f, cap) in enumerate(figs, 1):
    m = re.search(rf"Fig\. {i}\)", src)
    if not m:
        print("WARN: no citation for Fig.", i); continue
    cite = m.start(); para_end = src.index("\n\n", cite)
    ins = src.index("\n\n", para_end + 2) if src[para_end-1] == ":" else para_end
    block = f"\n\n![]({figdir}/{f}.png){{width=80%}}\n\n**Fig. {i}.** {cap}\n"
    src = src[:ins] + block + src[ins:]

pathlib.Path("/tmp/ei_build.md").write_text(src)
print("build markdown ready")
