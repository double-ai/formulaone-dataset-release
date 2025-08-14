# FormulaOne: Measuring the Depth of Algorithmic Reasoning Beyond Competitive Programming

This is the official repository for the paper:

**FormulaOne: Measuring the Depth of Algorithmic Reasoning Beyond Competitive Programming** <br>
*Gal Beniamini, Yuval Dor, Alon Vinnikov, Shir Granot Peled, Or Weinstein, Or Sharir, Noam Wies, Tomer Nussbaum, Nadav Schweiger, Ido Ben Shaul, Tomer Zekharya, Yoav Levine, Shai Shalev-Shwartz, Amnon Shashua* <br>
**AAI, July 2025**

Related resources:

- [FormulaOne Hugging Face space](https://huggingface.co/spaces/double-ai/FormulaOne-Leaderboard) including an **introduction** to the FormulaOne dataset and a **live leaderboard**.  
- FormulaOne [paper](https://arxiv.org/pdf/2507.13337).


## Repository Contents


* **FormulaOne Datasets**:
  * The `FormulaOne` core dataset, comprising 120 problems spanning Tier1 and Tier2. The dataset provided here includes the problem descriptions, and a **subset of the complete test suite**. The complete test suite is kept privately and is evaluated against when you submit to our leaderboard.
  * The `FormulaOne-Warmup` dataset, comprising 100 easier problems, including problem descriptions and the **complete test suite**. This dataset is released in full alongside a standalone evaluation environment for the community to train and evaluate on, supporting further research.
  
* **The Evaluation Framework**: The complete Python environment used to run and verify solutions against our comprehensive test suites. This can be found in the `/evaluation` directory.
* **Few-Shot Prompt**: The complete prompt, including three detailed examples, that was provided to the models during our evaluation. You can find this in the `/prompts` directory.
* **Example Solution**: A human-written reference solution for the classic `Dominating Set` problem, as mentioned in the paper, which demonstrates the expected structure of a valid submission. This is available in the `/examples` directory.


## Running the standalone evaluation environment

To run your own evaluation, refer to the `run.py` module provided in the `/evaluation` directory. This includes a demonstration of how to evaluate solutions against the test suites included in this repo.

## Project Setup

Install Python (recommended version: 3.13), then run the following commands to set up the environment.


```commandline
python -m venv /path/to/your/venv
pip install -r requirements.txt
```

## Licence
| Asset | Licence |
|---------------------------------------------------|---------|
| Evaluation source code (`/evaluator`) | Apache 2.0 |
| Problem statements &amp; public I/O (`/problems`) | CC BY 4.0 |

The accompanying NOTICE file must be preserved in any redistribution.
&quot;FormulaOne&quot; is used solely to identify this benchmark and may not be
used to imply endorsement of derivative works.
Privacy Notice: docs/privacy-policy.md