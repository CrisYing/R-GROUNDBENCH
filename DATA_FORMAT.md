# Data Format

R-GroundBench evaluators expect Hugging Face Dataset directories for benchmark inputs and JSONL files for model outputs. Paths below are relative to the repository root unless environment variables override them.

## VQA Input

Expected directory: `data/vqa_dataset`

Splits:

- `easy_basic`
- `medium_basic`
- `hard_basic`
- `easy_advanced`
- `medium_advanced`
- `hard_advanced`

Each VQA record contains the input scaffold, an instruction or property question, four answer options, the correct label, and metadata used by the evaluator.

Common fields:

- `split`: dataset split name.
- `question`: natural-language question.
- `instruction`: R-group edit instruction for Basic tasks.
- `original_esmiles`: scaffold E-SMILES.
- `original_image_path`: path to the rendered scaffold image.
- `option_A_smiles`, `option_B_smiles`, `option_C_smiles`, `option_D_smiles`: SMILES candidates.
- `option_A_image`, `option_B_image`, `option_C_image`, `option_D_image`: rendered candidate images when image options are available.
- `correct_answer`: one of `A`, `B`, `C`, or `D`.
- `property`: property category for Advanced tasks, when applicable.
- `trick`: evaluation subtype used for analysis, when applicable.

### NOTA Fields

NOTA means "None of the above." In NOTA records, option `D` can be the correct answer when options `A`, `B`, and `C` should be rejected.

Relevant fields:

- `is_none_of_above`: `true` if the record belongs to the NOTA evaluation subset.
- `nova_answer_is_d`: `true` when option `D` is the intended NOTA answer.
- `nova_type`: NOTA subtype, such as `type_b`, `type_c`, or `valid_d`.

## Generation Input

Expected directory: `data/generation_qa`

Splits:

- `easy`
- `hard`

Each generation record contains:

- `split` or `difficulty`: `easy` or `hard`.
- `original_esmiles`: input scaffold E-SMILES.
- `original_image_path`: path to the rendered scaffold image.
- `instruction`: R-group substitution instruction.
- `correct_smiles` or `output_smiles`: ground-truth edited molecule SMILES.
- `scaffold_smiles_clean`: scaffold used for scaffold-match evaluation.
- `trick`: evaluation subtype, when applicable.

## VQA Output

VQA predictions are stored as JSONL files under `results/vqa/<model>/`.

Each row records:

- `index`
- `split`
- `mode`
- `model`
- `correct_answer`
- `predicted`
- `raw_response`
- `is_correct`
- `elapsed`
- NOTA and property metadata when available

## Generation Output

Generation predictions are stored as JSONL files under `results/generation/<model>/`.

Each row records:

- `index`
- `split`
- `difficulty`
- `mode`
- `model`
- `ground_truth`
- `raw_response`
- `predicted_smiles`
- `is_valid`
- `is_exact_match`
- `tanimoto`
- `scaffold_match`
- `elapsed`

## External Baseline Predictions

Chemical-domain model predictions should be provided as JSONL files with stable record identifiers and predicted structures. For MarkushGrapher-2 scaffold predictions, use:

```bash
python evaluation/eval_mg2_pipeline.py --scaffold_mode mg2 --mg2_pred_dir path/to/predictions
```
