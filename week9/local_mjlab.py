from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


# Sửa path này thành path tới file task của bạn.
TASK_FILE = Path(__file__).parent / "my_cartpole_task.py"


def import_task_file(task_file: Path) -> None:
    """Import file task local để register_mjlab_task(...) được chạy."""
    task_file = task_file.resolve()

    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")

    spec = importlib.util.spec_from_file_location("local_mjlab_task", task_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import task file: {task_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def main() -> None:
    import_task_file(TASK_FILE)

    if len(sys.argv) < 2 or sys.argv[1] not in {"list", "train", "play"}:
        print(
            "Usage:\n"
            "  python local_mjlab.py list [keyword]\n"
            "  python local_mjlab.py train <TASK_ID> [mjlab train args...]\n"
            "  python local_mjlab.py play <TASK_ID> [mjlab play args...]\n"
        )
        raise SystemExit(2)

    command = sys.argv.pop(1)

    if command == "list":
        from mjlab.scripts.list_envs import main as mjlab_list_main

        mjlab_list_main()

    elif command == "train":
        from mjlab.scripts.train import main as mjlab_train_main

        mjlab_train_main()

    elif command == "play":
        from mjlab.scripts.play import main as mjlab_play_main

        mjlab_play_main()


if __name__ == "__main__":
    main()