import ast
from pathlib import Path
from types import SimpleNamespace

from robodojo.sim.environment.scene_manager.pose_restore import restore_saved_poses

ROOT = Path(__file__).resolve().parents[1]


class _PoseTarget:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def apply_saved_pose(self):
        self.events.append(self.name)


class _Sim:
    def __init__(self, events):
        self.events = events

    def sim_step(self, *, render):
        assert render is False
        self.events.append("step")


def test_saved_poses_restore_selected_environments_in_two_settled_phases():
    events = []
    tables = [_PoseTarget("table-0", events), _PoseTarget("table-1", events), _PoseTarget("table-2", events)]
    groups = [
        [{"rigid": _PoseTarget(f"rigid-{env_idx}", events)}, {"garment": _PoseTarget(f"garment-{env_idx}", events)}]
        for env_idx in range(3)
    ]

    restore_saved_poses([0, 2], tables, groups, _Sim(events), settle_steps=2)

    assert events == [
        "table-0",
        "table-2",
        "step",
        "step",
        "rigid-0",
        "garment-0",
        "rigid-2",
        "garment-2",
        "step",
        "step",
    ]


class _ChessboardParser:
    def is_A_xy_close_to_B_support_point(self, *, args):
        return args["label_A"] == "player_piece0" and args["B_tag"] == "cell/3" and args["threshold"] >= 0.05

    def is_A_point_above_B_point_by_z_range(self, *, args):
        return args["label_A"] == "player_piece0"


def test_loose_tic_tac_toe_piece_marks_cell_occupied_without_counting_as_placed():
    source_path = ROOT / "src/robodojo/sim/tasks/play_tic_tac_toe.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PlayTicTacToeCommon"
    )
    method_node = next(
        node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "check_chessboard_state"
    )
    namespace = {}
    exec(compile(ast.Module(body=[method_node], type_ignores=[]), str(source_path), "exec"), namespace)
    instance = SimpleNamespace(reward_manager=SimpleNamespace(func_parser=_ChessboardParser()))

    placed, empty_cells = namespace["check_chessboard_state"](instance, 0)

    assert placed == 0
    assert 3 not in empty_cells
    assert len(empty_cells) == 8
