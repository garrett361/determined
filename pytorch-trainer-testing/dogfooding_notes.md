# Dogfooding Notes

## Arguments of `.init()` and `.fit()`

General confusion about when config fields should be specified in yaml file versus in `.init()`
and `.fit()`.

If HPs specified in `.init()` and also in yaml file (on-cluster), the `.init()` hps are ignored. If
I am specifying HPS in `init()`, I would rather have those be the ones used (or at least a flag
giving me the option). Ideally, they'd be
merged: `hparams = {**hparams, ** cluster_info.trial.hparams}` or the opposite ordering, depending
on a flag.

## Bugs

- Setting `max_length` in units of records leads to never-ending experiments (Anda currently fixing)
  .

## Miscellaneous

### Validation Metrics Message/Stats

Seems likely unrelated to this pr, but I noticed that the validation metrics message can be
inaccurate in
its batch count through some off-by-one error.

For
instance, [this Trial](http://ec2-34-211-31-58.us-west-2.compute.amazonaws.com:8080/det/experiments/51/logs)
has a validation set of size 10 and is processed in
batches of size 1 (created with `miminal_tests/const_one_slot.yaml`), but the validation message
reads:

```bash
validated: 10 records in 0.00308s (3247.0 records/s), in 9 batches (2922.0 batches/s)
```

which doesn't make sense for any batch size (should be `in 10 batches`) and the `records/s`
and `batches/s` numbers should also match (they differ by a `9/10` factor).

### Training Units

`Record` unit is a little weird and seemingly not super useful? Probably no need to get rid of it,
though.

A slightly more useful unit would be `Step`, meaning actual optimizer steps. Numerically, this just
differs from `Batch`es by the gradient aggregation rate.

I fully support removing `records_per_epoch`. This means that users currently error out
if they specify training in terms of epochs, though. Intended behavior?

```
Failed to create experiment: invalid experiment configuration: version 0 experiment config is invalid: config is invalid: <config>.searcher.max_length: must specify
the top-level records_per_epoch when this field is in terms of epochs
```
