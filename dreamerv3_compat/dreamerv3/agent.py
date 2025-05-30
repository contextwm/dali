import embodied
import jax
import jax.numpy as jnp
import numpy as np
import ruamel.yaml as yaml
from collections import deque

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
from . import jaxutils
cast = jaxutils.cast_to_compute

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors, jaxagent, jaxutils, nets
from . import ninjax as nj


def nonzero_in(x, config):
    if x in config:
        if x:
            return True
    return False


@jaxagent.Wrapper
class Agent(nj.Module):

    configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "configs.yaml").read()
    )

    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = getattr(behaviors, config.task_behavior)(
            self.wm, self.act_space, self.config, name="task_behavior"
        )
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(
                self.wm, self.act_space, self.config, name="expl_behavior"
            )
        self.stack_len = config.batch_length
        self.num_envs = config.envs.amount
        self.img_encoder_dim = None

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
            jnp.zeros((self.num_envs, self.stack_len, self.obs_space["obs"].shape[0]), dtype=jnp.float32),
            jnp.zeros((self.num_envs, self.stack_len, self.act_space.shape[0]), dtype=jnp.float32),
            jnp.zeros((self.num_envs, self.stack_len, self.img_encoder_dim), dtype=jnp.float32),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    def policy(self, obs, state, mode="train"):
        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)

        (prev_latent, prev_action), task_state, expl_state, obs_stack, act_stack, embed_stack = state
        # input the latest obs at the first position shift the window 1 to the left
        obs_stack = obs_stack.at[:, 0].set(obs["obs"])
        obs_stack = jnp.roll(obs_stack, shift=-1, axis=1)
        act_stack = act_stack.at[:, 0].set(prev_action)
        act_stack = jnp.roll(act_stack, shift=-1, axis=1)

        embed = self.wm.encoder(obs)
        embed_stack = embed_stack.at[:, 0].set(embed)
        embed_stack = jnp.roll(embed_stack, shift=-1, axis=1)

        # ADD
        dcontext = None
        if self.wm.rssm._add_dcontext:
            dcontext = obs["context"]
        elif self.wm.use_ctx_encoder:
            dummy_len_dcontext = self.wm.ctx_encoder({
                "action": act_stack,
                "obs": obs_stack,
                "embed": embed_stack if 'embed' in self.config.ctx_encoder.inputs else None
            })
            dcontext = dummy_len_dcontext[:, -1]

        # OLD
        # dcontext = obs["context"] if self.wm.rssm._add_dcontext else None

        latent, _ = self.wm.rssm.obs_step(
            prev_latent,
            prev_action,
            embed,
            obs["is_first"],
            dcontext=dcontext,
        )

        self.expl_behavior.policy(latent, expl_state, dcontext=dcontext)
        task_outs, task_state = self.task_behavior.policy(
            latent, task_state, dcontext=dcontext
        )
        expl_outs, expl_state = self.expl_behavior.policy(
            latent, expl_state, dcontext=dcontext
        )
        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())

        state = ((latent, outs["action"]), task_state, expl_state, obs_stack, act_stack, embed_stack)
        return outs, state

    def train(self, data, state):
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        context = {**data, **wm_outs["post"]}

        # NOTE: Hacky solution #
        if self.img_encoder_dim is None:
            self.img_encoder_dim = wm_outs["embed"].shape[-1]

        # Overwrite start["context"] set in L:487
        if self.wm.use_ctx_encoder:
            ctx_enc_data = {
                "action": data["action"],
                "obs": data["obs"],
                "embed": wm_outs["embed"] if 'embed' in self.config.ctx_encoder.inputs else None,
            }
            context["context"] = self.wm.ctx_encoder(ctx_enc_data)
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)

        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        outs = {}
        return outs, state, metrics

    def report(self, data):
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key",):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs


class WorldModel(nj.Module):

    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        context_size = 0

        self.use_ctx_encoder = config.use_ctx_encoder
        if self.use_ctx_encoder:
            self.ctx_encoder = nets.CtxEncoder(**config.ctx_encoder, name="ctx_enc", rssm_config=config.rssm)
            # ctx encoder output size
            context_size = config.ctx_encoder["linear_ctx_out"]["hunits"]
        elif hasattr(config.rssm, "add_dcontext") and config.rssm.add_dcontext:
            context_size = obs_space["context"].shape[0]

        self.rssm = nets.RSSM(**config.rssm, context_size=context_size, name="rssm", add_enc_ctx=self.use_ctx_encoder)
        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }
        if hasattr(config, "use_context_head") and config.use_context_head:
            assert config.rssm.add_dcontext
            self.heads["context"] = nets.MLP(
                shapes["context"], **config.context_head, name="context"
            )
            assert "context" not in self.config.grad_heads

        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (state, outs, metrics) = self.opt(
            modules, self.loss, data, state, has_aux=True
        )
        return state, outs, metrics

    def loss(self, data, state):
        embed = self.encoder(data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate(
            [prev_action[:, None], data["action"][:, :-1]], 1
        )
        # ADD
        ctx = None
        if self.rssm._add_dcontext:
            ctx = data["context"]
        elif self.use_ctx_encoder:
            ctx_enc_data = {
                "action": data["action"],
                "obs": data["obs"],
                "embed": embed if 'embed' in self.config.ctx_encoder.inputs else None,
            }
            ctx = sg(self.ctx_encoder(ctx_enc_data))

        post, prior = self.rssm.observe(
            embed,
            prev_actions,
            data["is_first"],
            prev_latent,
            # ADD
            ctx,
            # OLD
            # data["context"] if self.rssm._add_dcontext else None,
        )
        dists = {}
        feats = {**post, "embed": embed}
        # add context to feats if add_context is True inline

        # OLD
        # if self.rssm._add_dcontext:
        #      feats["context"] = data["context"]

        # ADD
        if self.rssm._add_dcontext or self.use_ctx_encoder:
            feats["context"] = ctx

        # add to heads if add_context is True
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss
        # Forward model loss to shape context encoding
        if self.use_ctx_encoder:
            ctx_lossses = []
            for t in range(1, data["obs"].shape[1]):
                pad_width = data["obs"].shape[1] - t
                padded_obs = jnp.pad(
                    data["obs"][:, :t],
                    ((0, 0), (pad_width, 0), (0, 0)),
                    mode='edge'
                )
                padded_action = jnp.pad(
                    data["action"][:, :t],
                    ((0, 0), (pad_width, 0), (0, 0)),
                    mode='edge'
                )
                if 'embed' in self.config.ctx_encoder.inputs:
                    padded_embed = jnp.pad(
                        embed[:, :t],
                        ((0, 0), (pad_width, 0), (0, 0)),
                        mode='edge'
                    )
                else:
                    padded_embed = None

                ctx_enc_data = {
                    "action": padded_action,
                    "obs": padded_obs,
                    "embed": padded_embed
                }
                rolling_ctx = self.ctx_encoder(ctx_enc_data)

                is_first = cast(data["is_first"][:, t])
                prev_action = cast(prev_action)
                if self.rssm._action_clip > 0.0:
                    prev_action *= sg(
                        self.rssm._action_clip / jnp.maximum(self.rssm._action_clip, jnp.abs(prev_action))
                    )
                prev_state, prev_action = jax.tree_util.tree_map(
                    lambda x: self.rssm._mask(x, 1.0 - data["is_first"][:, t]), (prev_latent, prev_action)
                )
                prev_state = jax.tree_util.tree_map(
                    lambda x, y: x + self.rssm._mask(y, data["is_first"][:, t]),
                    prev_state,
                    self.rssm.initial(len(data["is_first"][:, t])),
                )

                # calculate h_t and prior (\hat{z}_t) based on prev state and action
                ctx_prior = self.rssm.img_step(sg(prev_state), prev_action, dcontext=sg(rolling_ctx[:, -1]))
                x = jnp.concatenate([sg(ctx_prior["deter"]), embed[:, t]], -1)
                # calculate z_t (posterior) based on prior h_t and embedding of o_t
                x = self.rssm.get("obs_out", nets.Linear, **self.rssm._kw)(x)
                stats = self.rssm._stats("obs_stats", x)
                dist = self.rssm.get_dist(stats)
                stoch = dist.sample(seed=nj.rng())
                ctx_post = {"stoch": stoch, "deter": ctx_prior["deter"], **stats}
                ctx_post, ctx_prior = cast(ctx_post), cast(ctx_prior)

                loss_fd = self.ctx_encoder.compute_loss({
                    **ctx_enc_data,
                    "context": rolling_ctx
                })
                if self.config.ctx_encoder.crossmodal:
                    loss_cross = self.ctx_encoder.reconstruct_wm_state_loss(
                        rolling_ctx[:, -1],
                        ctx_post["stoch"].reshape(ctx_post["stoch"].shape[0], -1)
                    )
                    ctx_lossses.append(loss_fd + self.config.ctx_encoder.lambda_cross * loss_cross)
                else:
                    ctx_lossses.append(loss_fd)

                prev_latent = ctx_post
                prev_action = data["action"][:, t]
            losses["dali"] = jnp.concatenate([jnp.stack(ctx_lossses, axis=1), jnp.zeros((data["obs"].shape[0], 1))], axis=-1)

        if hasattr(self.config, "use_context_head") and self.config.use_context_head:
            pure_context_head_fn = nj.pure(
                lambda ctx: self.heads["context"](ctx), nested=True
            )
            context_head_state = self.heads["context"].getm()
            adv_pred, _ = pure_context_head_fn(sg(context_head_state), nj.rng(), post)
            losses["context_adv"] = -adv_pred.log_prob(
                jnp.zeros_like(data["context"], jnp.float32)
            )
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())


        start = {
            k: v
            for k, v in start.items()
            if k in keys or ((self.rssm._add_dcontext or self.use_ctx_encoder) and k == "context")
        }

        start["action"] = policy(start)

        def step(prev, _):
            context_avail = "context" in prev
            prev = prev.copy()
            action = prev.pop("action")
            context = (
                prev.pop("context")
                if context_avail and (self.rssm._add_dcontext or self.use_ctx_encoder)
                else None
            )
            state = self.rssm.img_step(prev, action, dcontext=context)  # here
            state = {**state, "context": context} if context_avail else state
            return {**state, "action": policy(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])

        # ADD
        ctx = None
        if self.rssm._add_dcontext:
            ctx = data["context"]
        elif self.use_ctx_encoder:
            embed = self.encoder(data) if 'embed' in self.config.ctx_encoder.inputs else None
            ctx_enc_data = {
                "action": data["action"],
                "obs": data["obs"],
                "embed": embed if embed is not None else None
            }
            ctx = self.ctx_encoder(ctx_enc_data)

        context, _ = self.rssm.observe(
            self.encoder(data)[:6, :5],
            data["action"][:6, :5],
            data["is_first"][:6, :5],
            # ADD
            dcontext=ctx[:6, :5] if ctx is not None else None
            # OLD
            # dcontext=data["context"][:6, :5] if self.rssm._add_dcontext else None,
        )
        # add context to context dict
        # if self.rssm._add_dcontext:
        #     context["context"] = data["context"][:6, :5]
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](
            {
                **context,
                # OLD
                # "context": data["context"][:6, :5] if self.rssm._add_dcontext else None,
                # ADD
                "context": ctx[:6, :5] if ctx is not None else None,
            }
        )
        imagined = self.rssm.imagine(
            data["action"][:6, 5:],
            start,
            # OLD
            # dcontext=data["context"][:6, 5:] if self.rssm._add_dcontext else None,
            # ADD
            dcontext=ctx[:6, 5:] if ctx is not None else None,
        )
        imagined = (
            # OLD
            # {**imagined, "context": data["context"][:6, 5:]}
            # ADD
            {**imagined, "context": ctx[:6, 5:]}
            if self.rssm._add_dcontext or self.use_ctx_encoder
            else imagined
        )
        openl = self.heads["decoder"](imagined)
        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)
        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class ImagActorCritic(nj.Module):

    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        self.actor = nets.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {
            k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics
        }
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)
        self._add_dcontext = (
            hasattr(self.config.rssm, "add_dcontext") and self.config.rssm.add_dcontext
        )
        self.use_ctx_encoder = (
            hasattr(self.config, "use_ctx_encoder") and self.config.use_ctx_encoder
        )

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry, dcontext=None):
        # # if self.config.add_dcontext:
        # #     state["context"] = dcontext
        # if self._add_dcontext or self.use_ctx_encoder:
        #     state = {**state, "context": dcontext}
        #     # state["context"] = dcontext
        # return {"action": self.actor(state)}, carry
        return {
             "action": self.actor(
                 {**state, "context": dcontext} if self._add_dcontext or self.use_ctx_encoder else state
             )
         }, carry

    def train(self, imagine, start, context):
        # context includes the context hehe
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            # add
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        if self._add_dcontext:
            start["dcontext"] = context["context"]

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj, dcontext=None):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()
        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, "action"))
        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):

    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum(
                "...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs)
            )
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert (
            len(rew) == len(traj["action"]) - 1
        ), "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
