import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

from robodojo.core.layouts import resolve_layout_set
from robodojo.core.logging import configure_logging
from robodojo.core.models import SimulatorLaunchRequest
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles import load_environment_profile, load_scene_profile
from robodojo.core.scene_identity import require_matching_scene_identity
from robodojo.core.storage import assets_root

logger = logging.getLogger(__name__)

configure_logging()

MAX_INPROC_RESTARTS = 3

# AppLauncher inspects known arguments before registering its own --device
# option. Disable prefix matching so argparse does not mistake that option for
# RoboDojo's upstream-compatible --device_id during the inspection pass.
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--task_name", type=str)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--env_cfg_type",
    type=str,
    required=True,
    help="config file name for evaluation",
)
parser.add_argument(
    "--scene_config",
    type=str,
    default=None,
    help="resolved simulator scene config name",
)
parser.add_argument("--device_id", type=int, required=True, help="the device id for current process")
parser.add_argument(
    "--policy_name",
    type=str,
    required=True,
    help="XPolicyLab module name for deployment",
)
parser.add_argument("--port", type=int, required=True, help="the port for the policy WebSocket server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="IP address or hostname of the policy server. Defaults to localhost.",
)
parser.add_argument(
    "--protocol",
    choices=("ws",),
    default="ws",
    help=(
        "Env-to-policy transport. 'ws' is the default WebSocket protocol "
        "(msgpack frames over ws://host:port); also set as protocol: ws in deploy.yml."
    ),
)
parser.add_argument(
    "--policy_server_url",
    type=str,
    default="",
    help=(
        "Full WebSocket URL for the policy server (e.g. ws://127.0.0.1:9999). "
        "Built from --host and --port when omitted."
    ),
)
parser.add_argument(
    "--additional_info",
    type=str,
    required=True,
    help="additional information for the evaluation",
)


parser.add_argument("--seed", type=int, required=True, help="policy seed for eval")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _env_flag(name):
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


SCENE_EXPORT_REQUESTED = _env_flag("ROBODOJO_EXPORT_SCENE") or _env_flag("ROBODOJO_EXPORT_SCENE_ONLY")
SCENE_EXPORT_ONLY = _env_flag("ROBODOJO_EXPORT_SCENE_ONLY")
SCENE_EXPORT_LAYOUT_ID = int(os.environ.get("ROBODOJO_EXPORT_LAYOUT_ID", "0"))
SCENE_VISUAL_AUDIT_REQUESTED = _env_flag("ROBODOJO_SCENE_VISUAL_AUDIT")
if SCENE_VISUAL_AUDIT_REQUESTED and not SCENE_EXPORT_ONLY:
    raise ValueError("ROBODOJO_SCENE_VISUAL_AUDIT=1 is valid only with --export-scene-only")

# Safe to import before AppLauncher: env is a namespace package (no __init__)
# and GLOBAL_CONFIGS only imports os, so this pulls in no app-dependent code.
from robodojo.sim import tasks_registry
from robodojo.sim.environment.global_configs import ROOT_DIR
from robodojo.sim.launcher import resolve_scene_config

task_registry = tasks_registry


class PhysXBrokenError(Exception):
    pass


class PhysXFatalError(Exception):
    pass


class SceneExportError(RuntimeError):
    pass


def get_monitor():
    return None


REPOSITORY_PATHS = RepositoryPaths.resolve(ROOT_DIR)
ENVIRONMENT_PROFILE = load_environment_profile(REPOSITORY_PATHS, args_cli.env_cfg_type)
RESOLVED_SCENE_CONFIG = resolve_scene_config(
    REPOSITORY_PATHS,
    SimulatorLaunchRequest(
        task=args_cli.task_name,
        policy_name=args_cli.policy_name,
        host=args_cli.host,
        port=args_cli.port,
        env_config=args_cli.env_cfg_type,
        scene_config=args_cli.scene_config,
        additional_info=args_cli.additional_info,
    ),
    profile=ENVIRONMENT_PROFILE,
)
SCENE_PROFILE = load_scene_profile(REPOSITORY_PATHS, RESOLVED_SCENE_CONFIG)
RESOLVED_LAYOUT_SET = resolve_layout_set(
    config_root=REPOSITORY_PATHS.environment_configs,
    assets_root=assets_root(),
    benchmark="RoboDojo",
    layout_set=SCENE_PROFILE.document.layout_set,
    layout_source=SCENE_PROFILE.document.layout_source,
    task=args_cli.task_name,
    seed=args_cli.seed,
)


def _physx_monitor_needed(task_name) -> bool:
    """Enable the PhysX log monitor only for tasks whose Config declares a
    non-empty `Articulation` section (those bodies trigger the PhysX
    "Invalid PhysX transform" / CUDA failures we recover from). Read with a
    lightweight yaml load before AppLauncher; any failure falls back to
    enabled (fail-safe).
    """
    cfg_path = task_registry.task_config_path(REPOSITORY_PATHS.task_configs, task_name)
    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get("Articulation"))
    except Exception:
        return True


# Fix the run id for this entire eval invocation (across all in-process
# os.execv self-restarts and bash-level retries). When eval_policy.sh
# launches us it exports ROBODOJO_RUN_ID up-front; if a developer runs
# main.py directly we generate one here and propagate it via the
# environment so subsequent execv calls see the same value.
if not os.environ.get("ROBODOJO_RUN_ID"):
    os.environ["ROBODOJO_RUN_ID"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

enable_monitor = _physx_monitor_needed(args_cli.task_name)
logger.info("[main] PhysX monitor enabled=%s (task=%s)", enable_monitor, args_cli.task_name)
if enable_monitor:
    # Start before AppLauncher so Kit inherits the redirected stdout/stderr fds.
    from robodojo.sim.evaluation.physx_warning_monitor import (
        PhysXBrokenError,
        PhysXFatalError,
        get_monitor,
    )

    get_monitor().start(enabled=True)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robodojo.sim.scene_assets import prepare_scene_assets

PREPARED_SCENE_ASSETS = prepare_scene_assets(SCENE_PROFILE, args_cli.task_name)

from omegaconf import OmegaConf

from robodojo.core.storage import eval_work_root
from robodojo.sim.environment.global_configs import ENV_CONFIG_PATH
from robodojo.sim.evaluation.eval_env import create_eval_env
from robodojo.sim.utils.cluttered_generator import UnStableError
from robodojo.sim.utils.load_file import load_yaml
from robodojo.sim.utils.pipeline_utils import (
    process_config,
    process_randomization,
    resolve_random_task_num_envs,
)


def _eval_batch_from_deploy(policy_name):
    deploy_yml_path = os.path.join(ROOT_DIR, "XPolicyLab", "policy", policy_name, "deploy.yml")
    deploy_yml = load_yaml(deploy_yml_path) if os.path.isfile(deploy_yml_path) else {}
    return bool(deploy_yml.get("eval_batch", False))


def _resume_manifest_path(eval_cfg, run_id):
    """Mirror of EvalEnv.resume_manifest_path() so we can load BEFORE
    constructing the env. Keeping the layout aligned avoids drift between
    the writer and reader paths.
    """
    return os.path.join(
        str(eval_work_root()),
        eval_cfg["task_name"],
        eval_cfg["policy_name"],
        eval_cfg["config_name"],
        f"{eval_cfg.get('seed', 0)}_{eval_cfg.get('additional_info', '')}",
        f"_resume_{run_id}.json",
    )


def _load_resume_manifest(eval_cfg, run_id):
    """Return parsed manifest dict, or None if no resume is in progress."""
    path = _resume_manifest_path(eval_cfg, run_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as e:
        logger.warning("[main] failed to load resume manifest at %s: %s; ignoring.", path, e)
        return None
    require_matching_scene_identity(eval_cfg, data, context=f"resume manifest at {path}")
    logger.info(
        "[main] resuming from manifest %s (success=%s fail=%s completed=%s abandoned=%s restart_count=%s)",
        path,
        data.get("success_nums"),
        data.get("fail_nums"),
        len(data.get("completed_layout_ids") or []),
        len(data.get("abandoned_layout_ids") or []),
        data.get("restart_count", 0),
    )
    return data


def _delete_resume_manifest(env):
    """Best-effort cleanup at normal completion. Failure is non-fatal."""
    try:
        path = env.resume_manifest_path()
    except Exception:
        return
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info("[main] removed resume manifest %s (eval completed)", path)
    except Exception as e:
        logger.warning("[main] failed to unlink resume manifest %s: %s", path, e)


def _close_model_client(env):
    """Best-effort graceful close for policy communication."""
    try:
        model_client = getattr(env, "model_client", None)
        close = getattr(model_client, "close", None)
        if callable(close):
            close()
    except Exception as e:
        logger.warning("[main] failed to close model client: %s", e)


def _restart_or_exit(env, simulation_app, fatal_msg):
    """Persist progress and either os.execv-restart or sys.exit(99).

    Bounded by ROBODOJO_FATAL_RESTART_COUNT env var so a persistent hardware
    failure cannot put us in an infinite loop. The bash retry loop in
    eval_policy.sh provides a second layer of bounded restarts.
    """
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0")) + 1
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as e:
        logger.critical("persist_resume_manifest failed: %s", e)
    logger.critical(
        "PhysX kernel failure detected: %s; persisted manifest. In-process restart attempt %s/%s.",
        fatal_msg,
        restart_count,
        MAX_INPROC_RESTARTS,
    )
    try:
        simulation_app.close()
    except Exception:
        pass
    if restart_count <= MAX_INPROC_RESTARTS:
        os.environ["ROBODOJO_FATAL_RESTART_COUNT"] = str(restart_count)
        logger.critical("os.execv self-restart with run_id=%s", os.environ.get("ROBODOJO_RUN_ID"))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable] + sys.argv)
    logger.critical(
        "in-process restart cap reached (%s); exiting with rc=99 for bash-level retry.", MAX_INPROC_RESTARTS
    )
    sys.exit(99)


def _exit_for_shell_restart(env, fatal_msg):
    """Persist progress, then let eval_policy.sh restart a fresh process."""
    restart_count = int(os.environ.get("ROBODOJO_FATAL_RESTART_COUNT", "0"))
    try:
        env.persist_resume_manifest(restart_count=restart_count)
    except Exception as e:
        logger.critical("persist_resume_manifest failed: %s", e)
    logger.critical("PhysX requested shell-level restart: %s; exiting with rc=99 for bash-level retry.", fatal_msg)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(99)


def main():
    """Assemble the env config, build the eval env, and run the eval loop with
    PhysX crash/resume recovery until the requested episode count is reached.
    """
    task_name = args_cli.task_name
    num_envs = 1 if SCENE_EXPORT_ONLY else args_cli.num_envs
    eval_cfg = load_yaml(ENVIRONMENT_PROFILE.path)
    eval_cfg["scene_config"] = RESOLVED_SCENE_CONFIG
    eval_cfg["scene_component"] = SCENE_PROFILE.document.component
    eval_cfg["scene_profile_hash"] = SCENE_PROFILE.identity_hash
    eval_cfg["layout_config_name"] = SCENE_PROFILE.document.layout_set
    eval_cfg["layout_source"] = SCENE_PROFILE.document.layout_source
    eval_cfg["layout_set_hash"] = RESOLVED_LAYOUT_SET.identity_hash
    eval_cfg["scene_asset_hash"] = PREPARED_SCENE_ASSETS.identity_hash
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = num_envs
    eval_cfg["device_id"] = args_cli.device_id
    eval_batch = False if SCENE_EXPORT_ONLY else _eval_batch_from_deploy(args_cli.policy_name)
    eval_cfg["eval_batch"] = eval_batch
    eval_cfg["policy_name"] = args_cli.policy_name
    eval_cfg["additional_info"] = args_cli.additional_info
    eval_cfg["seed"] = args_cli.seed
    eval_cfg["physx_monitor_enabled"] = enable_monitor

    deploy_cfg = {}
    deploy_cfg["policy_name"] = args_cli.policy_name
    deploy_cfg["port"] = args_cli.port
    deploy_cfg["host"] = args_cli.host
    deploy_cfg["protocol"] = args_cli.protocol
    deploy_cfg["policy_server_url"] = args_cli.policy_server_url or f"ws://{args_cli.host}:{args_cli.port}"
    deploy_cfg["evaluation_id"] = os.environ["ROBODOJO_RUN_ID"]
    deploy_cfg["trial_id"] = f"{task_name}-{os.environ['ROBODOJO_RUN_ID']}"
    deploy_cfg["action_case_id"] = f"{task_name}_case"
    deploy_cfg["repeat_index"] = None
    env_cfg = OmegaConf.create(
        {
            "sim": load_yaml(os.path.join(ENV_CONFIG_PATH, "sim", eval_cfg["config"]["sim"] + ".yml")),
            "scene": load_yaml(SCENE_PROFILE.component_path),
            "camera": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "camera",
                    eval_cfg["config"]["camera"] + ".yml",
                )
            ),
            "robot": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "robot",
                    eval_cfg["config"]["robot"] + ".yml",
                )
            ),
            "task_env": load_yaml(task_registry.task_config_path(REPOSITORY_PATHS.task_configs, task_name)),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    # TaskEnv defaults a missing sim.device to CPU. Use AppLauncher's resolved
    # process-local device so the environment and Kit select the same GPU.
    OmegaConf.update(env_cfg, "sim.device", args_cli.device, force_add=True)
    capped_num_envs = resolve_random_task_num_envs(task_name, num_envs, env_cfg.sim)
    if capped_num_envs != num_envs:
        logger.info("[main] Random task %s: num_envs capped %s -> %s ", task_name, num_envs, capped_num_envs)
    num_envs = capped_num_envs
    if not eval_batch and num_envs != 1:
        logger.info(
            "[main] eval_batch=false in XPolicyLab/policy/%s/deploy.yml; forcing num_envs %s -> 1",
            args_cli.policy_name,
            num_envs,
        )
        num_envs = 1
    eval_cfg["num_envs"] = num_envs
    OmegaConf.update(env_cfg, "sim.scene.num_envs", num_envs, force_add=True)
    OmegaConf.update(env_cfg, "eval_cfg.num_envs", num_envs, force_add=True)
    env_cfg = process_randomization(env_cfg)
    env_cfg, eval_num = process_config(
        env_cfg,
        task_name=task_name,
        resolved_scene_config=RESOLVED_SCENE_CONFIG,
    )

    if os.environ.get("EVAL_NUM"):
        _env_eval_num = os.environ.get("EVAL_NUM")
        if str(_env_eval_num).lower() != "native":
            eval_num = min(int(_env_eval_num), int(eval_num))
    eval_cfg["eval_num"] = eval_num

    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )

    env_cfg.sim.seed = [0 for _ in range(num_envs)]
    run_id = os.environ["ROBODOJO_RUN_ID"]
    resume_state = None if SCENE_EXPORT_ONLY else _load_resume_manifest(eval_cfg, run_id)
    env = create_eval_env(
        env_cfg,
        simulation_app,
        resume_state=resume_state,
        policy_enabled=not SCENE_EXPORT_ONLY,
    )
    if SCENE_EXPORT_REQUESTED and SCENE_EXPORT_LAYOUT_ID not in env.seed_manager.seed_info:
        raise ValueError(
            f"layout id {SCENE_EXPORT_LAYOUT_ID} is unavailable for task={task_name} "
            f"seed={args_cli.seed}; available={sorted(env.seed_manager.seed_info)}"
        )
    if SCENE_EXPORT_REQUESTED and resume_state is None and not SCENE_EXPORT_ONLY:
        # Preserve the evaluator's full seed set while making the requested
        # snapshot layout the first reset/rollout and avoiding a later duplicate.
        env.seed_manager.seed_list.remove(SCENE_EXPORT_LAYOUT_ID)
        env.seed_manager.seed_list.insert(0, SCENE_EXPORT_LAYOUT_ID)
    eval_time = env.success_nums + env.fail_nums
    if SCENE_EXPORT_ONLY:
        env.env_seeds = [SCENE_EXPORT_LAYOUT_ID]
    elif eval_time >= eval_num:
        # Already complete on resume - nothing left to do.
        env.env_seeds = None
    else:
        env.env_seeds = env.seed_manager.get_seeds(max_count=eval_num - eval_time)
    export_pending = SCENE_EXPORT_REQUESTED
    while env.env_seeds is not None:
        retry_round = False
        if enable_monitor:
            get_monitor().reset()
        bad_envs = None
        try:
            env.reset(seed=env.env_seeds)
            matched_replay_dir = os.environ.get("ROBODOJO_MATCHED_REPLAY_DIR")
            if matched_replay_dir:
                from robodojo.sim.calibration.matched_replay import run_matched_state_replay

                manifest = ENVIRONMENT_PROFILE.matched_replay_manifest
                if manifest is None:
                    raise ValueError(
                        f"environment profile {ENVIRONMENT_PROFILE.name} does not declare a matched replay manifest"
                    )
                report = run_matched_state_replay(
                    env,
                    manifest,
                    Path(matched_replay_dir),
                )
                logger.info("[matched-replay] wrote %s", report)
            if export_pending:
                from robodojo.sim.scene_export.exporter import export_scene_snapshot

                export_dir = os.environ.get("ROBODOJO_EXPORT_SCENE_DIR") or os.path.join(env.save_dir, "scene_snapshot")
                try:
                    exported_scene = export_scene_snapshot(env, export_dir, SCENE_EXPORT_LAYOUT_ID)
                    if SCENE_VISUAL_AUDIT_REQUESTED:
                        from robodojo.sim.scene_export.visual_audit import run_scene_visual_audit

                        run_scene_visual_audit(env, exported_scene, SCENE_EXPORT_LAYOUT_ID)
                except Exception as e:
                    raise SceneExportError(str(e)) from e
                export_pending = False
                if SCENE_EXPORT_ONLY:
                    env.seed_manager.eval_step()
                    _close_model_client(env)
                    env.close()
                    simulation_app.close()
                    logger.info("[scene-export] scene-only mode complete; policy rollout skipped")
                    return
            env.run_eval()
            env.seed_manager.eval_step()

        except SceneExportError:
            import traceback

            traceback.print_exc()
            _close_model_client(env)
            env.close()
            simulation_app.close()
            raise

        except PhysXFatalError as e:
            # Unrecoverable: GPU/CUDA context is dead. Persist progress
            # and re-exec (or sys.exit(99) for bash to restart).
            if not enable_monitor:
                raise
            if get_monitor().requires_shell_restart():
                _exit_for_shell_restart(env, str(e))
            _restart_or_exit(env, simulation_app, str(e))

        except PhysXBrokenError as e:
            # Monitor caught the warning in time.
            bad_envs = sorted(e.broken_envs)

        except UnStableError:
            env.seed_manager.eval_step()
        except Exception as e:
            import traceback

            logger.warning("[Eval] unhandled exception during reset/run_eval: %s: %s", type(e).__name__, e)
            traceback.print_exc()
            if enable_monitor and get_monitor().is_fatal():
                fatal_msg = get_monitor().get_fatal_message() or str(e)
                if get_monitor().requires_shell_restart():
                    _exit_for_shell_restart(env, fatal_msg)
                _restart_or_exit(env, simulation_app, fatal_msg)
            bad = {i for i in get_monitor().get_broken_envs() if i < env.num_envs} if enable_monitor else set()
            if bad:
                logger.info(
                    "[PhysX] downstream exception %s with monitor broken_envs=%s; treating as PhysX break.",
                    type(e).__name__,
                    sorted(bad),
                )
                bad_envs = sorted(bad)
            else:
                env.seed_manager.eval_step()

        if bad_envs is not None:
            # Abandon broken-env seeds, refill from the seed queue, and
            # retry this round iff there is at least one real seed left.
            bad_seeds = env.get_seeds_for_envs(bad_envs)
            env.abandoned_seeds.update(bad_seeds)

            replacements = env.seed_manager.get_seeds(max_count=len(bad_envs)) or []
            bad_env_set = set(bad_envs)
            new_batch = [None] * env.num_envs
            for env_idx, seed in env.current_env_seed_map.items():
                if env_idx not in bad_env_set:
                    new_batch[env_idx] = seed
            for k, env_idx in enumerate(bad_envs):
                if k < len(replacements):
                    new_batch[env_idx] = replacements[k]
            env.env_seeds = new_batch

            real_remaining = sum(1 for s in env.env_seeds if s is not None)
            logger.info(
                "[PhysX] broken envs=%s -> abandon seeds=%s; refill from queue=%s; new batch=%s; real_remaining=%s",
                bad_envs,
                sorted(bad_seeds),
                replacements,
                env.env_seeds,
                real_remaining,
            )
            env.close()
            if real_remaining == 0:
                logger.info("[PhysX] no real seeds remaining in this batch, advancing.")
                retry_round = False
            else:
                retry_round = True

        if retry_round:
            continue

        logger.info(
            "Success nums: %s, Fail nums: %s, Unstable nums: %s", env.success_nums, env.fail_nums, env.unstable_nums
        )
        eval_time = env.success_nums + env.fail_nums
        if eval_time >= eval_num:
            break

        env.env_seeds = env.seed_manager.get_seeds(max_count=eval_num - eval_time)
        if env.env_seeds is None:
            logger.info("No more seeds to run, exiting.")
            break

        env.close()

    _delete_resume_manifest(env)
    _close_model_client(env)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
