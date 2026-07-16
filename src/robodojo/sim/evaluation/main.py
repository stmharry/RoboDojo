import argparse
from datetime import datetime
import logging
import os
from pathlib import Path

import warp as _project_warp

EXPECTED_WARP_VERSION = "1.11.0"
_loaded_warp_version = getattr(_project_warp, "__version__", None)
if _loaded_warp_version != EXPECTED_WARP_VERSION:
    raise RuntimeError(
        "RoboDojo must preload the project-pinned Warp "
        f"{EXPECTED_WARP_VERSION} before Isaac Sim; loaded {_loaded_warp_version!r} "
        f"from {getattr(_project_warp, '__file__', '<unknown>')}. "
        "Run the simulator through the locked robodojo entrypoint."
    )

# Import only after the version guard. AppLauncher otherwise prepends Isaac
# Sim's bundled Warp 1.8.2 before cuRobo is imported.
from isaaclab.app import AppLauncher

from robodojo.core.asset_identity import inspect_environment_assets
from robodojo.core.experiments.selection import compose_experiment, resolve_recipe
from robodojo.core.layouts import resolve_layout_set
from robodojo.core.logging import configure_logging
from robodojo.core.paths import RepositoryPaths
from robodojo.core.profiles.environment import load_environment_profile
from robodojo.core.profiles.scene import load_scene_profile, validate_scene_environment_compatibility
from robodojo.core.revisions import git_revision
from robodojo.core.storage import assets_root
from robodojo.core.workspace import validate_resolved_layout_set
from robodojo.sim.evaluation.communication import close_model_client as _close_model_client
from robodojo.sim.evaluation.completion import ensure_evaluation_complete
from robodojo.sim.evaluation.configuration import build_deploy_config, build_evaluation_config
from robodojo.sim.evaluation.restart import (
    exit_for_shell_restart as _exit_for_shell_restart,
    restart_or_exit as _restart_or_exit,
)
from robodojo.sim.evaluation.resume import (
    delete_resume_manifest as _delete_resume_manifest,
    load_resume_manifest as _load_resume_manifest,
)

logger = logging.getLogger(__name__)

configure_logging()
logger.info("Using project-pinned Warp %s from %s", _loaded_warp_version, _project_warp.__file__)

# AppLauncher inspects known arguments before registering its own --device
# option. Disable prefix matching so argparse does not mistake that option for
# RoboDojo's upstream-compatible --device_id during the inspection pass.
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--task-protocol", type=str, required=True)
parser.add_argument("--episode-horizon", type=int, required=True)
parser.add_argument("--evaluation-episodes", type=int, required=True)
parser.add_argument("--recipe", type=str, required=True)
parser.add_argument("--experiment-hash", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--environment",
    type=str,
    required=True,
    help="config file name for evaluation",
)
parser.add_argument(
    "--scene",
    type=str,
    required=True,
    help="resolved simulator scene config name",
)
parser.add_argument("--device_id", type=int, required=True, help="the device id for current process")
parser.add_argument(
    "--policy_name",
    type=str,
    required=True,
    help="XPolicyLab module name for deployment",
)
parser.add_argument("--policy_profile", type=str, required=True, help="RoboDojo policy profile name")
parser.add_argument("--policy_descriptor_hash", type=str, required=True)
parser.add_argument(
    "--policy_reference_match",
    choices=("reference_match", "domain_shift", "unspecified"),
    required=True,
)
parser.add_argument("--port", type=int, required=True, help="the port for the policy WebSocket server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="IP address or hostname of the policy server. Defaults to localhost.",
)
parser.add_argument(
    "--transport",
    choices=("ws",),
    default="ws",
    help=(
        "Env-to-policy transport. 'ws' is the default WebSocket transport "
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
FIRST_FRAME_CAPTURE_REQUESTED = _env_flag("ROBODOJO_CAPTURE_FIRST_FRAME")
FIRST_FRAME_CAPTURE_DIR = os.environ.get("ROBODOJO_FIRST_FRAME_DIR", "").strip()
SIMULATOR_ONLY = SCENE_EXPORT_ONLY or FIRST_FRAME_CAPTURE_REQUESTED
if SCENE_VISUAL_AUDIT_REQUESTED and not SCENE_EXPORT_ONLY:
    raise ValueError("ROBODOJO_SCENE_VISUAL_AUDIT=1 is valid only with --export-scene-only")
if FIRST_FRAME_CAPTURE_REQUESTED and SCENE_EXPORT_ONLY:
    raise ValueError("first-frame capture cannot be combined with scene-export-only mode")
if FIRST_FRAME_CAPTURE_REQUESTED and not FIRST_FRAME_CAPTURE_DIR:
    raise ValueError("ROBODOJO_CAPTURE_FIRST_FRAME=1 requires ROBODOJO_FIRST_FRAME_DIR")

# Safe to import before AppLauncher: env is a namespace package (no __init__)
# and GLOBAL_CONFIGS only imports os, so this pulls in no app-dependent code.
from robodojo.sim import task_discovery
from robodojo.sim.environment.global_configs import ROOT_DIR

task_registry = task_discovery


class PhysXBrokenError(Exception):
    pass


class PhysXFatalError(Exception):
    pass


class SceneExportError(RuntimeError):
    pass


class FirstFrameCaptureError(RuntimeError):
    pass


def get_monitor():
    return None


REPOSITORY_PATHS = RepositoryPaths.resolve(ROOT_DIR)
RUNTIME_CONTRACT = None
if args_cli.policy_profile == "manual":
    ENVIRONMENT_PROFILE = load_environment_profile(REPOSITORY_PATHS, args_cli.environment)
    SCENE_PROFILE = load_scene_profile(REPOSITORY_PATHS, args_cli.scene)
    validate_scene_environment_compatibility(SCENE_PROFILE, ENVIRONMENT_PROFILE)
else:
    RUNTIME_CONTRACT = (
        resolve_recipe(REPOSITORY_PATHS, args_cli.recipe)
        if args_cli.recipe != "manual"
        else compose_experiment(
            REPOSITORY_PATHS,
            policy_name=args_cli.policy_profile,
            environment_name=args_cli.environment,
            scene_name=args_cli.scene,
            task_protocol=args_cli.task_protocol,
        )
    )
    expected = (
        RUNTIME_CONTRACT.protocol.task,
        RUNTIME_CONTRACT.task_protocol,
        RUNTIME_CONTRACT.protocol.episode_horizon,
        RUNTIME_CONTRACT.protocol.evaluation_episodes,
        RUNTIME_CONTRACT.environment.name,
        RUNTIME_CONTRACT.scene.name,
        RUNTIME_CONTRACT.policy_descriptor_hash,
        RUNTIME_CONTRACT.policy_reference_match,
        RUNTIME_CONTRACT.identity_hash,
        (REPOSITORY_PATHS.root / RUNTIME_CONTRACT.policy.policy_dir).name,
    )
    actual = (
        args_cli.task,
        args_cli.task_protocol,
        args_cli.episode_horizon,
        args_cli.evaluation_episodes,
        args_cli.environment,
        args_cli.scene,
        args_cli.policy_descriptor_hash,
        args_cli.policy_reference_match,
        args_cli.experiment_hash,
        args_cli.policy_name,
    )
    if actual != expected:
        raise ValueError(f"simulator experiment arguments {actual} do not match resolved experiment {expected}")
    ENVIRONMENT_PROFILE = RUNTIME_CONTRACT.environment
    SCENE_PROFILE = RUNTIME_CONTRACT.scene
RESOLVED_SCENE_CONFIG = SCENE_PROFILE.name
ROBODOJO_REVISION = git_revision(REPOSITORY_PATHS.root)
XPOLICYLAB_REVISION = git_revision(REPOSITORY_PATHS.xpolicy_root)
RESOLVED_LAYOUT_SET = resolve_layout_set(
    config_root=REPOSITORY_PATHS.environment_configs,
    assets_root=assets_root(),
    benchmark="RoboDojo",
    layout_set=SCENE_PROFILE.document.layout_set,
    layout_source=SCENE_PROFILE.document.layout_source,
    task=args_cli.task,
    seed=args_cli.seed,
)
validate_resolved_layout_set(
    RESOLVED_LAYOUT_SET,
    task_config_path=REPOSITORY_PATHS.task_configs / f"{args_cli.task}.yml",
    workspace=ENVIRONMENT_PROFILE.document.workspace,
    robot_config_path=ENVIRONMENT_PROFILE.component_paths["robot"],
)


def _physx_monitor_needed(task_name) -> bool:
    """Enable the PhysX log monitor only for tasks whose Config declares a
    non-empty `Articulation` section (those bodies trigger the PhysX
    "Invalid PhysX transform" / CUDA failures we recover from). Read with a
    lightweight yaml load before AppLauncher; any failure falls back to
    enabled (fail-safe).
    """
    cfg_path = REPOSITORY_PATHS.task_configs / f"{task_name}.yml"
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

enable_monitor = _physx_monitor_needed(args_cli.task)
logger.info("[main] PhysX monitor enabled=%s (task=%s)", enable_monitor, args_cli.task)
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

from robodojo.sim.scene_assets import inspect_scene_assets

PREPARED_SCENE_ASSETS = inspect_scene_assets(SCENE_PROFILE, args_cli.task)
PREPARED_ENVIRONMENT_ASSETS = inspect_environment_assets(ENVIRONMENT_PROFILE)

from omegaconf import OmegaConf

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


def main():
    """Assemble the env config, build the eval env, and run the eval loop with
    PhysX crash/resume recovery until the requested episode count is reached.
    """
    task_name = args_cli.task
    num_envs = 1 if SIMULATOR_ONLY else args_cli.num_envs
    eval_cfg = build_evaluation_config(
        environment=ENVIRONMENT_PROFILE,
        scene=SCENE_PROFILE,
        layout_set=RESOLVED_LAYOUT_SET,
        environment_assets=PREPARED_ENVIRONMENT_ASSETS,
        scene_assets=PREPARED_SCENE_ASSETS,
        runtime_experiment=RUNTIME_CONTRACT,
        task=task_name,
        task_protocol=args_cli.task_protocol,
        episode_horizon=args_cli.episode_horizon,
        evaluation_episodes=args_cli.evaluation_episodes,
        recipe=args_cli.recipe,
        experiment_hash=args_cli.experiment_hash,
        policy_name=args_cli.policy_name,
        policy_profile=args_cli.policy_profile,
        seed=args_cli.seed,
        additional_info=args_cli.additional_info,
        device_id=args_cli.device_id,
        num_envs=num_envs,
        physx_monitor_enabled=enable_monitor,
        robodojo_revision=ROBODOJO_REVISION,
        xpolicylab_revision=XPOLICYLAB_REVISION,
        assets_root=assets_root(),
    )
    eval_batch = False if SIMULATOR_ONLY else _eval_batch_from_deploy(args_cli.policy_name)
    eval_cfg["eval_batch"] = eval_batch
    deploy_cfg = build_deploy_config(
        policy_name=args_cli.policy_name,
        host=args_cli.host,
        port=args_cli.port,
        transport=args_cli.transport,
        server_url=args_cli.policy_server_url,
        run_id=os.environ["ROBODOJO_RUN_ID"],
        task_protocol=args_cli.task_protocol,
    )
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
            "scene_mounts": SCENE_PROFILE.document.mounts.model_dump(mode="python", exclude_none=True),
            "task_env": load_yaml(REPOSITORY_PATHS.task_configs / f"{task_name}.yml"),
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
        evaluation_episodes=args_cli.evaluation_episodes,
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
    resume_state = None if SIMULATOR_ONLY else _load_resume_manifest(eval_cfg, run_id)
    env = create_eval_env(
        env_cfg,
        simulation_app,
        resume_state=resume_state,
        policy_enabled=not SIMULATOR_ONLY,
        record_video_enabled=not FIRST_FRAME_CAPTURE_REQUESTED,
    )
    if (SCENE_EXPORT_REQUESTED or FIRST_FRAME_CAPTURE_REQUESTED) and (
        SCENE_EXPORT_LAYOUT_ID not in env.seed_manager.seed_info
    ):
        raise ValueError(
            f"layout id {SCENE_EXPORT_LAYOUT_ID} is unavailable for task={task_name} "
            f"seed={args_cli.seed}; available={sorted(env.seed_manager.seed_info)}"
        )
    if SCENE_EXPORT_REQUESTED and resume_state is None and not SIMULATOR_ONLY:
        # Preserve the evaluator's full seed set while making the requested
        # snapshot layout the first reset/rollout and avoiding a later duplicate.
        env.seed_manager.seed_list.remove(SCENE_EXPORT_LAYOUT_ID)
        env.seed_manager.seed_list.insert(0, SCENE_EXPORT_LAYOUT_ID)
    eval_time = env.success_nums + env.fail_nums
    if SIMULATOR_ONLY:
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
            if FIRST_FRAME_CAPTURE_REQUESTED:
                from robodojo.sim.scene_export.first_frame import capture_first_frame

                try:
                    capture_first_frame(env, FIRST_FRAME_CAPTURE_DIR, SCENE_EXPORT_LAYOUT_ID)
                except Exception as e:
                    raise FirstFrameCaptureError(str(e)) from e
                env.seed_manager.eval_step()
                _close_model_client(env)
                env.close()
                simulation_app.close()
                logger.info("[first-frame] capture complete; policy rollout skipped")
                return
            env.run_eval()
            env.seed_manager.eval_step()

        except (SceneExportError, FirstFrameCaptureError):
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
    if not SIMULATOR_ONLY:
        ensure_evaluation_complete(eval_time=eval_time, requested=eval_num)


if __name__ == "__main__":
    main()
