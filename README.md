# Dynamics-Aligned Latent Imagination (DALI)

![counterfactual1](./counterfactual_dim6.gif)
![counterfactual2](./counterfactual_dim3.gif)

## Setup
Use `uv` to setup your python environment.

```bash
uv sync
uv pip install -e ./dreamerv3_compat
uv pip install -e ./
```

For ball-in-cup, you need to install the local `./CARL` provided.
```bash
uv pip install -e ./CARL
```

## Training

Run scripts in `./local_scripts/` to generate results for experts, random policies and DALI variants.

## Record Data for Anaylsis

```bash
uv run -m contextual_mbrl.dreamer.record_context --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337
```

### Log data for counterfactual dreams
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility --logdir logs/carl_dmc_ball_in_cup_double_box_enc_img_dec_img_ctxencoder_transformer_grssm_normalized/1337 --jax.platform cpu
```

### Log data for imagined counterfactual obs trajectories
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility_obs --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337 --jax.platform cpu
```

### Record dataset for counterfactual obs analysis
```bash
uv run -m contextual_mbrl.dreamer.record_counterfactual_plausibility_obs_dataset --logdir logs/carl_dmc_walker_double_box_enc_img_dec_img_ctxencoder_transformer_normalized/1337 --jax.platform cpu
```

## Plots

To generate the plots, run the scripts in the `./analysis` directory.
