"""Run DroidRun against AndroidWorld benchmarks.

This adapter borrows the AndroidWorld compatibility layer from a working local
benchmark runner and swaps the execution backend to DroidRun's Python SDK.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import pkgutil
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PORTAL_PACKAGE = "com.droidrun.portal"
ANDROID_WORLD_A11Y_COMPONENT = (
    "com.google.androidenv.accessibilityforwarder/"
    "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder"
)
DROIDRUN_PORTAL_A11Y_COMPONENT = (
    f"{PORTAL_PACKAGE}/com.droidrun.portal.service.DroidrunAccessibilityService"
)
DEFAULT_DROIDRUN_REPO = Path(__file__).resolve().parents[1]


def _adb(*args: str, device: Optional[str], timeout: int = 30) -> str:
    cmd = ["adb"] + (["-s", device] if device else []) + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return ""
    return result.stdout.strip()


def _enabled_a11y_services(device: Optional[str]) -> str:
    return _adb(
        "shell",
        "settings get secure enabled_accessibility_services",
        device=device,
    )


def _sync_device_time(device: Optional[str]) -> bool:
    """Best-effort sync of emulator wall clock to current UTC time."""
    timestamp = datetime.now(timezone.utc).strftime("%m%d%H%M%Y.%S")
    subprocess.run(
        ["adb"] + (["-s", device] if device else []) + ["root"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    _adb("shell", f"date -u {timestamp}", device=device, timeout=15)
    current = _adb("shell", "date", device=device, timeout=15)
    restored = str(datetime.now(timezone.utc).year) in current
    print(f"    Time sync: synced={restored} device_time={current}")
    return restored


def _wait_for_accessibility(device: Optional[str], timeout_sec: int = 20) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        dumpsys = _adb("shell", "dumpsys accessibility", device=device, timeout=15)
        if "com.droidrun.portal.service.DroidrunAccessibilityService" in dumpsys:
            return True
        time.sleep(1)
    return False


def _restore_accessibility(device: Optional[str]) -> bool:
    services = f"{ANDROID_WORLD_A11Y_COMPONENT}:{DROIDRUN_PORTAL_A11Y_COMPONENT}"
    _adb(
        "shell",
        f"cmd appops set {PORTAL_PACKAGE} ACCESS_RESTRICTED_SETTINGS allow",
        device=device,
        timeout=15,
    )
    _adb(
        "shell",
        f"settings put secure enabled_accessibility_services '{services}'",
        device=device,
        timeout=15,
    )
    _adb(
        "shell",
        "settings put secure accessibility_enabled 1",
        device=device,
        timeout=15,
    )
    _adb(
        "shell",
        f"monkey -p {PORTAL_PACKAGE} -c android.intent.category.LAUNCHER 1",
        device=device,
        timeout=20,
    )
    restored = _wait_for_accessibility(device)
    print(
        f"    Accessibility restore: restored={restored} "
        f"enabled={_enabled_a11y_services(device)}"
    )
    return restored


def _import_android_world():
    try:
        base_agent = importlib.import_module("android_world.agents.base_agent")
        suite_utils = importlib.import_module("android_world.suite_utils")
        env_launcher = importlib.import_module("android_world.env.env_launcher")
        try:
            runner = importlib.import_module("android_world.run_lib")
        except ModuleNotFoundError:
            runner = importlib.import_module("android_world.episode_runner")
        return base_agent, suite_utils, env_launcher, runner
    except Exception as exc:
        print("ERROR: failed to import AndroidWorld runtime pieces.")
        print(f"IMPORT ERROR: {exc}")
        traceback.print_exc()
        return None


def _infer_console_port(device_serial: Optional[str]) -> Optional[int]:
    if not device_serial:
        return 5554
    match = re.fullmatch(r"emulator-(\d+)", device_serial)
    if not match:
        return None
    return int(match.group(1))


def _load_env(env_launcher, device_serial: Optional[str]):
    console_port = _infer_console_port(device_serial)
    candidates = []
    if console_port is not None:
        candidates.append({"console_port": console_port, "adb_path": "adb"})
        candidates.append({"console_port": console_port})
    candidates.append({"adb_path": "adb"})
    candidates.append({})

    last_error = None
    for kwargs in candidates:
        try:
            return env_launcher.load_and_setup_env(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(
        f"Could not call load_and_setup_env with known signatures: {last_error}"
    )


def _run_episode(runner, agent, task, env, max_steps: int):
    goal = (
        getattr(task, "goal", None)
        or getattr(task, "task_goal", None)
        or getattr(task, "instruction", None)
        or getattr(task, "prompt", None)
    )
    if goal is None and hasattr(task, "params"):
        params = getattr(task, "params")
        if isinstance(params, dict):
            goal = params.get("goal") or params.get("instruction") or params.get("prompt")
    if goal is None:
        raise RuntimeError(
            f"Could not extract goal text from task instance of type {type(task).__name__}"
        )

    candidates = [
        lambda: runner.run_episode(goal, agent, max_n_steps=max_steps),
        lambda: runner.run_episode(goal=goal, agent=agent, max_n_steps=max_steps),
        lambda: runner.run_episode(goal, agent, max_n_steps=max_steps, print_fn=print),
        lambda: runner.run_episode(goal=goal, agent=agent, max_n_steps=max_steps, print_fn=print),
    ]
    last_error = None
    for call in candidates:
        try:
            return call()
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not call run_episode with known signatures: {last_error}")


def _prepare_task(task, env) -> None:
    if getattr(task, "_droidrun_prepared", False):
        return

    last_type_error = None
    prepared = False
    for method_name in ("initialize_task", "initialize", "setup", "set_up"):
        method = getattr(task, method_name, None)
        if not callable(method):
            continue
        for call in (lambda m=method: m(env), lambda m=method: m()):
            try:
                call()
                prepared = True
                break
            except TypeError as exc:
                last_type_error = exc
        if prepared:
            break

    if not prepared and last_type_error is not None:
        raise RuntimeError(
            f"Could not initialize task {type(task).__name__}: {last_type_error}"
        )

    try:
        task._droidrun_prepared = True
    except Exception:
        pass


def _verify_task_success(task, env) -> Optional[bool]:
    verifier = getattr(task, "is_successful", None)
    if not callable(verifier):
        return None

    if hasattr(task, "initialized"):
        try:
            task.initialized = True
        except Exception:
            pass

    last_type_error = None
    for call in (lambda: verifier(env), lambda: verifier()):
        try:
            result = call()
            if isinstance(result, tuple):
                result = result[0]
            if isinstance(result, (int, float)):
                return result > 0
            return bool(result)
        except TypeError as exc:
            last_type_error = exc
        except Exception as exc:
            raise RuntimeError(
                f"AndroidWorld verifier failed for {type(task).__name__}: {exc}"
            ) from exc

    raise RuntimeError(
        f"Could not call verifier for {type(task).__name__}: {last_type_error}"
    )


def _run_suite(runner, agent, task_registry, env, max_steps: int):
    if hasattr(runner, "run_suite"):
        candidates = [
            lambda: runner.run_suite(agent, task_registry, env),
            lambda: runner.run_suite(agent=agent, task_registry=task_registry, env=env),
            lambda: runner.run_suite(agent=agent, tasks=task_registry, env=env),
        ]
        last_error = None
        for call in candidates:
            try:
                return call()
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(f"Could not call run_suite with known signatures: {last_error}")

    if hasattr(runner, "run_episode"):
        results = {}
        if isinstance(task_registry, dict):
            iterable = task_registry.items()
        else:
            iterable = enumerate(task_registry)

        for index, entry in enumerate(iterable):
            if isinstance(task_registry, dict):
                task_name, task_instances = entry
            else:
                task_name = f"task_{index}"
                task_instances = entry[1]

            if isinstance(task_instances, list):
                instance_results = []
                for task_instance in task_instances:
                    _prepare_task(task_instance, env)
                    episode_result = _run_episode(runner, agent, task_instance, env, max_steps)
                    verified_success = _verify_task_success(task_instance, env)
                    print(f"    AndroidWorld verifier: success={verified_success}")
                    instance_results.append(
                        {
                            "episode_result": episode_result,
                            "verified_success": verified_success,
                        }
                    )
                results[task_name] = (
                    instance_results[0] if len(instance_results) == 1 else instance_results
                )
            else:
                _prepare_task(task_instances, env)
                episode_result = _run_episode(runner, agent, task_instances, env, max_steps)
                verified_success = _verify_task_success(task_instances, env)
                print(f"    AndroidWorld verifier: success={verified_success}")
                results[task_name] = {
                    "episode_result": episode_result,
                    "verified_success": verified_success,
                }
        return results

    raise RuntimeError("AndroidWorld runner module does not expose run_suite or run_episode")


def _result_success(value) -> bool:
    if isinstance(value, list):
        return all(_result_success(item) for item in value)
    if isinstance(value, dict) and "verified_success" in value:
        return bool(value["verified_success"])
    step_data = getattr(value, "step_data", None)
    if isinstance(step_data, dict):
        success_values = step_data.get("success")
        if isinstance(success_values, list) and success_values:
            return bool(success_values[-1])
        if isinstance(success_values, bool):
            return success_values
    data = getattr(value, "data", None)
    if isinstance(data, dict) and "success" in data:
        return bool(data["success"])
    if isinstance(value, dict) and "success" in value:
        return bool(value["success"])
    return bool(value)


def _count_results(results) -> tuple[int, int]:
    def _flatten(value):
        if isinstance(value, dict):
            if "verified_success" in value:
                return [value]
            flattened = []
            for item in value.values():
                flattened.extend(_flatten(item))
            return flattened
        if isinstance(value, list):
            flattened = []
            for item in value:
                flattened.extend(_flatten(item))
            return flattened
        return [value]

    values = _flatten(results)
    passed = sum(1 for value in values if _result_success(value))
    return passed, len(values)


def _find_task_registry_dict(suite_utils):
    task_eval_module = importlib.import_module("android_world.task_evals.task_eval")
    task_eval_base = getattr(task_eval_module, "TaskEval", None)
    if task_eval_base is None:
        return None

    def _is_task_registry(value):
        if not isinstance(value, dict) or not value:
            return False
        if not all(isinstance(key, str) for key in value.keys()):
            return False
        for item in value.values():
            if not isinstance(item, type):
                return False
            if not issubclass(item, task_eval_base):
                return False
        return True

    if _is_task_registry(getattr(suite_utils, "TASK_REGISTRY", None)):
        return getattr(suite_utils, "TASK_REGISTRY")

    android_world_pkg = importlib.import_module("android_world")
    for module_info in pkgutil.walk_packages(
        android_world_pkg.__path__, android_world_pkg.__name__ + "."
    ):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for name in dir(module):
            if name.startswith("__"):
                continue
            value = getattr(module, name)
            if _is_task_registry(value):
                return value
    return None


def _find_task_eval_classes():
    task_eval_module = importlib.import_module("android_world.task_evals.task_eval")
    task_eval_base = getattr(task_eval_module, "TaskEval", None)
    if task_eval_base is None:
        return {}

    android_world_pkg = importlib.import_module("android_world")
    discovered = {}
    for module_info in pkgutil.walk_packages(
        android_world_pkg.__path__, android_world_pkg.__name__ + "."
    ):
        try:
            module = importlib.import_module(module_info.name)
        except Exception:
            continue
        for name in dir(module):
            if name.startswith("_"):
                continue
            value = getattr(module, name)
            if not isinstance(value, type):
                continue
            if value is task_eval_base or not issubclass(value, task_eval_base):
                continue
            if name in {
                "GenericTaskEval",
                "MiniWoBTask",
                "TestableMiniWoBTaskForTest",
                "MockTaskEval",
                "FakeTaskEval",
            }:
                continue
            discovered[name] = value
    return discovered


def _create_suite(suite_utils, selected_tasks, n_task_combinations: int):
    seed = 42
    registry_dict = _find_task_registry_dict(suite_utils)
    if registry_dict is not None:
        if selected_tasks:
            missing = [name for name in selected_tasks if name not in registry_dict]
            if missing:
                fallback_classes = _find_task_eval_classes()
                matched_fallback = {
                    name: fallback_classes[name] for name in missing if name in fallback_classes
                }
                if matched_fallback:
                    registry_dict = dict(registry_dict)
                    registry_dict.update(matched_fallback)
        return suite_utils.create_suite(
            registry_dict,
            n_task_combinations=n_task_combinations,
            seed=seed,
            tasks=selected_tasks,
        )

    if selected_tasks:
        fallback_classes = _find_task_eval_classes()
        filtered = {
            name: fallback_classes[name] for name in selected_tasks if name in fallback_classes
        }
        missing = [name for name in selected_tasks if name not in filtered]
        if missing:
            sample = ", ".join(sorted(fallback_classes.keys())[:20])
            raise RuntimeError(
                f"Could not locate AndroidWorld task(s): {missing}. "
                f"Sample discovered TaskEval classes: {sample}"
            )
        return suite_utils.create_suite(
            filtered,
            n_task_combinations=n_task_combinations,
            seed=seed,
            tasks=selected_tasks,
        )

    return suite_utils.create_suite(
        suite_utils.ANDROID_WORLD_TASKS,
        n_task_combinations=n_task_combinations,
        seed=seed,
    )


def _bootstrap_droidrun_repo(repo_path: Path) -> None:
    if not repo_path.exists():
        raise FileNotFoundError(f"DroidRun repo not found: {repo_path}")
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


@dataclass
class DroidRunOptions:
    config_path: Optional[str]
    device_serial: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    temperature: Optional[float]
    base_url: Optional[str]
    api_base: Optional[str]
    vision: Optional[bool]
    reasoning: Optional[bool]
    agent_steps: Optional[int]
    debug: Optional[bool]


async def _run_droidrun_goal(goal: str, options: DroidRunOptions) -> dict[str, Any]:
    from droidrun import DroidAgent, load_llm
    from droidrun.config_manager.loader import ConfigLoader

    config = ConfigLoader.load(options.config_path)

    if options.device_serial is not None:
        config.device.serial = options.device_serial
    if options.agent_steps is not None:
        config.agent.max_steps = options.agent_steps
    if options.reasoning is not None:
        config.agent.reasoning = options.reasoning
    if options.vision is not None:
        config.agent.manager.vision = options.vision
        config.agent.executor.vision = options.vision
        config.agent.fast_agent.vision = options.vision
    if options.debug is not None:
        config.logging.debug = options.debug

    llm = None
    if options.provider or options.model:
        if not (options.provider and options.model):
            raise ValueError("Both --provider and --model are required together")
        llm_kwargs: dict[str, Any] = {}
        if options.temperature is not None:
            llm_kwargs["temperature"] = options.temperature
        if options.base_url is not None:
            llm_kwargs["base_url"] = options.base_url
        if options.api_base is not None:
            llm_kwargs["api_base"] = options.api_base
        llm = load_llm(options.provider, model=options.model, **llm_kwargs)

    agent = DroidAgent(
        goal=goal,
        llms=llm,
        config=config,
        timeout=1000,
        runtype="developer",
    )
    handler = agent.run()
    result = await handler
    return {
        "success": bool(result.success),
        "reason": result.reason,
        "steps": result.steps,
    }


def _build_droidrun_android_world_agent(base_agent):
    class DroidRunAndroidWorldAgent(base_agent.EnvironmentInteractingAgent):
        def __init__(
            self,
            env,
            options: DroidRunOptions,
            name: str = "DroidRun",
        ):
            super().__init__(env, name=name)
            self._options = options
            self._counter = 0
            self._result_type = getattr(base_agent, "AgentInteractionResult", None)

        def step(self, goal: str):
            self._counter += 1
            print(f"\n-> DroidRun executing: {goal[:120]}...")
            start = time.time()
            result = asyncio.run(_run_droidrun_goal(goal, self._options))
            elapsed = time.time() - start
            print(
                "    DroidRun result: "
                f"success={result['success']} steps={result['steps']} elapsed={elapsed:.1f}s"
            )
            if result["reason"]:
                print(f"    Reason: {result['reason']}")

            data = {
                "goal": goal,
                "success": result["success"],
                "reason": result["reason"],
                "steps": result["steps"],
                "elapsed_seconds": round(elapsed, 2),
            }
            if self._result_type is None:
                return {"done": True, "data": data}
            return self._result_type(done=True, data=data)

        def reset(self, start_on_home_screen: bool = False) -> None:
            self._counter = 0
            if start_on_home_screen:
                _adb("shell", "input keyevent KEYCODE_HOME", device=self._options.device_serial)
                time.sleep(1)
            _sync_device_time(self._options.device_serial)
            _restore_accessibility(self._options.device_serial)

    return DroidRunAndroidWorldAgent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DroidRun on AndroidWorld benchmark")
    parser.add_argument(
        "--droidrun-repo",
        default=str(DEFAULT_DROIDRUN_REPO),
        help=f"Path to the local DroidRun repo (default: {DEFAULT_DROIDRUN_REPO})",
    )
    parser.add_argument("--config", help="DroidRun config.yaml path")
    parser.add_argument("--device", help="ADB device serial")
    parser.add_argument("--provider", help="Override LLM provider")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--temperature", type=float, help="Override model temperature")
    parser.add_argument("--base-url", help="Override LLM base_url")
    parser.add_argument("--api-base", help="Override LLM api_base")
    parser.add_argument("--vision", action="store_true", help="Enable vision on all agents")
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable manager/executor reasoning mode",
    )
    parser.add_argument(
        "--agent-steps",
        type=int,
        default=15,
        help="DroidRun max steps per benchmark goal (default: 15)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DroidRun debug logging")
    parser.add_argument(
        "--task",
        action="append",
        help="Single AndroidWorld task name; may be passed multiple times",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="AndroidWorld task names to run (default: all)",
    )
    parser.add_argument(
        "--n_task_combinations",
        type=int,
        default=3,
        help="Task combinations per AndroidWorld task (default: 3)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum AndroidWorld episode steps per task (default: 10)",
    )
    parser.add_argument(
        "--results-json",
        help="Optional path to write raw benchmark results as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _bootstrap_droidrun_repo(Path(args.droidrun_repo).expanduser())
    android_world_modules = _import_android_world()
    if android_world_modules is None:
        return 1
    base_agent, suite_utils, env_launcher, run_lib = android_world_modules
    DroidRunAndroidWorldAgent = _build_droidrun_android_world_agent(base_agent)

    selected_tasks = args.tasks or args.task
    options = DroidRunOptions(
        config_path=args.config,
        device_serial=args.device,
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        api_base=args.api_base,
        vision=True if args.vision else None,
        reasoning=True if args.reasoning else None,
        agent_steps=args.agent_steps,
        debug=True if args.debug else None,
    )

    print("Setting up AndroidWorld environment...")
    env = _load_env(env_launcher, args.device)
    _sync_device_time(args.device)
    _restore_accessibility(args.device)

    agent = DroidRunAndroidWorldAgent(env=env, options=options)
    task_registry = _create_suite(
        suite_utils,
        selected_tasks,
        n_task_combinations=args.n_task_combinations,
    )

    print(f"Running {len(task_registry)} task entries...")
    results = _run_suite(run_lib, agent, task_registry, env, args.max_steps)
    passed, total = _count_results(results)

    if args.results_json:
        output_path = Path(args.results_json).expanduser()
        output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"Saved raw results to {output_path}")

    print(f"\n{'=' * 50}")
    print(f"DroidRun AndroidWorld Score: {passed}/{total} = {passed / total * 100:.1f}%")
    print(f"{'=' * 50}")

    if hasattr(env, "close"):
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
