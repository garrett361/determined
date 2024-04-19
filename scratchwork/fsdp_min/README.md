# FSDP + Core API

Example of using FSDP with determined and Core API. (Relatively) simple transformer model adapted from [GPT-fast
](https://github.com/pytorch-labs/gpt-fast) training on fake data.

Tested:

- Does not error on the pytorch 2.0 image specified in `config.yaml` when running on T4s.
- Checkpoint save and reload.
- Metric reporting.

Not tested:

- Everything else

To run:

```bash
det e create config.yaml .
```
