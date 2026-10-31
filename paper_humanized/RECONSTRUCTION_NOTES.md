# Reconstruction notes

The original LaTeX source was not available, so this project was rebuilt from the compiled PDF (`main.pdf`, 6 pages).

## How to use on Overleaf
1. Upload this ZIP (New Project → Upload Project).
2. Set Menu → Compiler to **pdfLaTeX**. `IEEEtran.cls` and `IEEEtran.bst` are built into Overleaf, so they are not bundled here.
3. Recompile (pdfLaTeX → BibTeX → pdfLaTeX ×2 runs automatically).

## What is reproduced as printed
- Title, authors, affiliations, emails, abstract, index terms, all section and subsection text, and section order.
- The observation vector, Eqs. (1)–(3), and all parameter values.
- Tables I–III: every number as printed.
- All figure captions.
- Fig. 1: recreated in TikZ from the vector drawing, with the same blocks, labels, arrows, and colours.
- Figs. 2–5: lossless PNGs extracted from the PDF at their embedded resolution (199–418 ppi). Higher quality needs the original plot files.
- 20 references in the original order; numbering [1]–[20] is reproduced by citation order.

## Deviations from the original PDF
| Item | Original | Here | Why |
|---|---|---|---|
| Hyphenation artifacts | e.g. "bursthistory", "perprocess" | "burst-history", "per-process" | Line-break artifacts from PDF extraction |
| Name in Related Work | "Ontáñón" | "Ontañón" | Spelling of the author's name |
| Ref [2] | "Costa-juárez et al., sb3-contrib…, 2022" | Raffin et al., JMLR 22(268), 2021, with the sb3-contrib URL in a note | The original author could not be matched; this is the verified maintainers' citation. The original entry is preserved as a comment in `refs.bib` and can be swapped back. |
| Ref [12] | "S. Huang and S. O. nón" | "S. Huang and S. Ontañón", plus DOI | Verified at doi:10.32473/flairs.v35i.130584 |
| Fonts and headings | XeLaTeX build (Latin Modern look, normal-case headings) | pdfLaTeX IEEEtran (Times, small-caps headings) | Standard IEEE conference output. Page breaks and float positions may differ slightly. |
| Table font | — | `\footnotesize` | Needed for Tables I–III to fit the column |

Refs [13] and [14] are kept exactly as printed, with verified alternatives noted in comments in `refs.bib`.

## Not changed: open issues for the authors
These are documented in `FINDINGS.md` §5. The most important:
1. Realistic-workload numbers in Table I don't reconcile with Table II, Fig. 5, and the text (+0.6 vs −0.05; −0.7 vs −0.28; 10.91 vs 10.68).
2. Table III doesn't say which workload it uses.
3. SRTF is called an "upper bound" in Sec. IV and a "lower bound" in the Fig. 2 caption.
4. The Fig. 3 caption says P4 exceeded the aging threshold, but the image shows WAIT 0.
5. The Fig. 4 caption says three edge families are shown, but the image shows two, with no ready processes.
6. The dashboard images carry their own "FIGURE 1/2" titles.
7. W1 ∈ ℝ^{32×20} vs 8 features per node.
8. "Starvation rate" is not defined in the text.

## Verification done
The project was test-compiled locally against a stand-in class, because the real IEEEtran is not installed in the build environment. All sections, equations, three tables, five figures, and 20 bibliography entries rendered, with 0 errors, 0 undefined references, and 0 overfull boxes. Final page layout must be confirmed on Overleaf with the real IEEEtran class.
