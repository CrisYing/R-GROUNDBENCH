# Prompt Templates

## VQA

The VQA evaluator combines a scaffold input, an instruction or property question, and four options. Depending on mode, the scaffold and options may be images or SMILES strings.

## Generation

The generation evaluator asks the model to apply one R-group edit instruction to a Markush scaffold and return a tagged SMILES string.

Expected output format:

```text
<smiles>CCO</smiles>
```
