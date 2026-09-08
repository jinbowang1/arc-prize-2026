#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""持久工作台的单元测试 —— 跨轮保留模型自己写的函数与笔记。

对照 Retrodict 全通轨迹的真实条件: 它有持久 workspace(自建 arclog.py 解析库 +
playbook.md 经验手册), 我们的沙箱每次调用新建临时目录、代码不留。本测试锁住
"函数跨轮活着 + 笔记跨轮活着"这两件事。

跑: python3 tests/test_workbench.py
"""
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "build_v15_worldmodel/src/ARC3-Inference"
sys.path.insert(0, str(SRC))

from inference.agent.python_tool_sandbox import run_sandboxed_python  # noqa: E402

NOOP = lambda acts: {"executed": False}  # noqa: E731


def run(code, **kw):
    return run_sandboxed_python(code=code, timeout_seconds=15, initial_state={},
                                action_handler=NOOP, **kw)


class TestWorkbenchPersistence(unittest.TestCase):

    def test_function_survives_to_next_turn(self):
        """第一轮定义的函数, 第二轮还能调用。"""
        r1 = run("def lattice(c, r):\n    return (6 + 8 * c, 8 + 8 * r)\nresult = lattice(1, 1)")
        self.assertEqual(r1.get("result"), [14, 16], r1.get("error"))
        wb = r1.get("workbench_source")
        self.assertTrue(wb and "def lattice" in wb, "第一轮没有把函数存进工作台")

        r2 = run("result = lattice(2, 3)", workbench_source=wb)
        self.assertFalse(r2.get("error"), r2.get("error"))
        self.assertEqual(r2.get("result"), [22, 32], "第二轮调不到上一轮的函数")

    def test_notes_survive_to_next_turn(self):
        """笔记本(playbook)跨轮保留, 且模型能往上追加。"""
        r1 = run("notes = '- confirmed: clicking a button toggles its north neighbor'")
        self.assertIn("north neighbor", r1.get("notes") or "", "第一轮没捕获 notes")

        r2 = run("notes = notes + '\\n- confirmed: palette is binary [11, 14]'",
                 notes=r1["notes"])
        self.assertIn("north neighbor", r2.get("notes") or "", "第二轮丢了旧笔记")
        self.assertIn("binary", r2.get("notes") or "", "第二轮没存下新笔记")

    def test_notes_readable_when_absent(self):
        """从没写过笔记时, notes 也要存在(空串), 别让模型撞 NameError。"""
        r = run("result = len(notes)")
        self.assertFalse(r.get("error"), r.get("error"))
        self.assertEqual(r.get("result"), 0)

    def test_redefining_replaces_old_version(self):
        """同名函数以最新一轮为准, 别让旧版本压住新版本。"""
        r1 = run("def solve():\n    return 'v1'\nresult = solve()")
        r2 = run("def solve():\n    return 'v2'\nresult = solve()",
                 workbench_source=r1.get("workbench_source"))
        r3 = run("result = solve()", workbench_source=r2.get("workbench_source"))
        self.assertEqual(r3.get("result"), "v2", "同名函数没被新版覆盖")

    def test_broken_function_does_not_kill_the_turn(self):
        """工作台里带着语法坏掉的历史代码时, 本轮仍然要能跑。"""
        r = run("result = 1 + 1", workbench_source="def broken(:\n    pass")
        self.assertFalse(r.get("error"), "坏的历史代码把本轮打挂了")
        self.assertEqual(r.get("result"), 2)

    def test_world_model_channel_still_works(self):
        """原有的世界模型通道不能被工作台改动破坏。"""
        r = run("def world_model(before_frame, action):\n    return before_frame")
        self.assertTrue(r.get("world_model_source"), "world_model 捕获被打断了")

    def test_notes_are_size_capped(self):
        """笔记有上限, 免得把 prompt 撑爆。"""
        r = run("notes = 'x' * 100000")
        self.assertLessEqual(len(r.get("notes") or ""), 8192, "notes 没有上限")



class TestWorkbenchReachesTheModel(unittest.TestCase):
    """工作台存下来还不够 —— 模型每轮必须在提示词里看见它。

    这一层是昨晚栽过的地方: 沙箱到宿主到提示词是三段链路, 任何一段断了都是
    "数据生成了但模型没看到"。
    """

    def _agent(self, **attrs):
        from inference.agent.tool_agent import ToolAgent
        a = ToolAgent.__new__(ToolAgent)
        a._summarized_knowledge = {}
        a._workbench_source = None
        a._notes = None
        for k, v in attrs.items():
            setattr(a, k, v)
        return a

    def test_function_names_appear_in_prompt(self):
        a = self._agent(_workbench_source="def lattice(c, r):\n    return (c, r)\n\ndef solve():\n    return 1")
        text = "\n".join(a._summarized_knowledge_lines())
        self.assertIn("lattice()", text, "工作台里的函数没出现在提示词里")
        self.assertIn("solve()", text)

    def test_notes_appear_in_prompt(self):
        a = self._agent(_notes="- confirmed: palette is binary [11, 14]")
        text = "\n".join(a._summarized_knowledge_lines())
        self.assertIn("binary [11, 14]", text, "笔记没回到提示词里")

    def test_silent_when_workbench_empty(self):
        a = self._agent()
        self.assertEqual(a._summarized_knowledge_lines(), [], "工作台空时不该往提示词里塞东西")

    def test_broken_workbench_does_not_break_prompt(self):
        a = self._agent(_workbench_source="def broken(:")
        self.assertEqual(a._summarized_knowledge_lines(), [], "坏代码把提示词组装打挂了")

class TestNumpyAvailable(unittest.TestCase):
    """64x64 网格分析没有 numpy 等于用手数格子 —— 全通方有, 我们必须有。"""

    def test_numpy_imports_in_sandbox(self):
        r = run("import numpy as np\nresult = int(np.zeros((64, 64), dtype=int).sum())")
        self.assertFalse(r.get("error"), r.get("error"))
        self.assertEqual(r.get("result"), 0)

    def test_numpy_real_work(self):
        r = run("import numpy as np\n"
                "b = np.arange(16).reshape(4, 4)\n"
                "result = [int(b[b > 10].sum()), [int(x) for x in np.unique(b % 3)]]")
        self.assertFalse(r.get("error"), r.get("error"))
        self.assertEqual(r.get("result"), [65, [0, 1, 2]])

    def test_blocked_modules_still_blocked(self):
        """开 numpy 不等于开门 —— 危险模块仍要挡住。"""
        for mod in ("socket", "subprocess", "shutil"):
            r = run("import %s" % mod)
            self.assertTrue(r.get("error"), "%s 竟然能 import" % mod)


class TestSwitchIsConsistent(unittest.TestCase):
    """开关必须一刀切干净 —— 09-08 的 bug: 关闭组的提示词照样宣传 numpy, 而那组
    numpy 被挡, 模型一 import 就报错白烧步数, 直接把基线组的分数压掉一半。
    提示词说有的能力, 运行时就必须真有。"""

    def _prompt_text(self, on):
        import os
        import subprocess
        import sys
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from inference.agent import prompts\n"
                "print(prompts.PYTHON_ADDENDUM + prompts.COMPACT_TOOL_SESSION_ADDENDUM)" % str(SRC))
        env = dict(os.environ, ARC3_WORKBENCH=on)
        return subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, env=env).stdout

    def test_numpy_advertised_only_when_available(self):
        for on in ("1", "0"):
            advertised = "numpy" in self._prompt_text(on)
            import os
            env = dict(os.environ, ARC3_WORKBENCH=on)
            import subprocess, sys
            code = ("import sys; sys.path.insert(0, %r)\n"
                    "from inference.agent.python_tool_sandbox import run_sandboxed_python as R\n"
                    "r = R(code='import numpy', timeout_seconds=10, initial_state={},"
                    " action_handler=lambda a: {})\n"
                    "print('OK' if not r.get('error') else 'BLOCKED')" % str(SRC))
            actual = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                    text=True, env=env).stdout.strip()
            available = (actual == "OK")
            self.assertEqual(advertised, available,
                             "ARC3_WORKBENCH=%s: 提示词宣传 numpy=%s 但运行时可用=%s"
                             % (on, advertised, available))

    def test_workbench_wording_matches_switch(self):
        self.assertIn("workbench persists", self._prompt_text("1"))
        self.assertIn("Snippets are not saved", self._prompt_text("0"))
        self.assertNotIn("workbench persists", self._prompt_text("0"))


class TestBoardHelpers(unittest.TestCase):
    """预置解析库必须在沙箱里现成可用 —— 模型不必自己写。

    依据 09-08 实测: 24 局里 19 局的模型从没定义过一个可复用函数, 所以"让模型
    自己攒工具"这条路走不通; 全通方靠的也是作者写好的 arclog.py。
    """

    def test_helpers_defined_without_model_writing_them(self):
        # 沙箱的 builtins 受限(没有 globals), 直接引用: 少一个就 NameError
        r = run("result = [callable(x) for x in (grid, at, find, counts, diff, crop, objects, moved)]")
        self.assertFalse(r.get("error"), r.get("error"))
        self.assertEqual(r.get("result"), [True] * 8, "预置的 8 个 helper 没全注入")

    def test_helpers_are_callable(self):
        r = run("result = [callable(grid), callable(diff), callable(objects)]")
        self.assertEqual(r.get("result"), [True, True, True], r.get("error"))

    def test_helpers_off_by_switch(self):
        import os, subprocess, sys
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from inference.agent.python_tool_sandbox import run_sandboxed_python as R\n"
                "r = R(code='result = callable(grid)', timeout_seconds=10, initial_state={},"
                " action_handler=lambda a: {})\n"
                "print('GONE' if 'NameError' in (r.get('error') or '') else 'STILL_THERE')" % str(SRC))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=dict(os.environ, ARC3_HELPERS="0")).stdout.strip()
        self.assertEqual(out, "GONE", "ARC3_HELPERS=0 时 helper 仍然存在")

    def test_numpy_available_for_helpers_even_without_workbench(self):
        """grid() 依赖 numpy —— 关掉工作台也必须还能用, 否则 helper 是残的。"""
        import os, subprocess, sys
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from inference.agent.python_tool_sandbox import run_sandboxed_python as R\n"
                "r = R(code='import numpy; result = 1', timeout_seconds=10, initial_state={},"
                " action_handler=lambda a: {})\n"
                "print('OK' if not r.get('error') else 'BLOCKED')" % str(SRC))
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=dict(os.environ, ARC3_WORKBENCH="0", ARC3_HELPERS="1")).stdout.strip()
        self.assertEqual(out, "OK", "关工作台后 numpy 被挡, helper 的 grid() 会废掉")


if __name__ == "__main__":
    unittest.main(verbosity=2)
