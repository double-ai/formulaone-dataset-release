# FormulaOne: Measuring the Depth of Algorithmic Reasoning Beyond Competitive Programming

This is the official repository for the paper:

**FormulaOne: Measuring the Depth of Algorithmic Reasoning Beyond Competitive Programming** <br>
*Gal Beniamini, Yuval Dor, Alon Vinnikov, Shir Granot Peled, Or Weinstein, Or Sharir, Noam Wies, Tomer Nussbaum, Ido Ben Shaul, Tomer Zekharya, Yoav Levine, Shai Shalev-Shwartz, Amnon Shashua* <br>
**AAI, July 2025**

FormulaOne is a new benchmark designed to challenge frontier AI models. The benchmark is constructed from a vast and conceptually diverse family of dynamic programming problems derived from Monadic Second-Order (MSO) logic on graphs, a framework with profound connections to theoretical computer science.

## Repository Contents

### Currently Available

* **Few-Shot Prompt**: The complete prompt, including three detailed examples, that was provided to the models during our evaluation. You can find this in the `/prompts` directory.
* **Example Solution**: A human-written reference solution for the classic `Dominating Set` problem, as mentioned in the paper, which demonstrates the expected structure of a valid submission. This is available in the `/examples` directory.

### Arriving Within a Few Days

* **The Full Datasets**: The complete `FormulaOne` (120 problems) and `FormulaOne-Warmup` (100 problems) datasets.
* **The Evaluation Framework**: The complete Python environment used to run and verify solutions against our comprehensive test suites.
