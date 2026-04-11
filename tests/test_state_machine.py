"""Unit tests for AgentStateMachine.

测试 AgentStateMachine 状态机的各种状态转换逻辑。
"""

import pytest
from src.state_machine import AgentStateMachine, InvalidTransitionError
from src.models import AgentState


class TestAgentStateMachine:
    """Test cases for AgentStateMachine class."""

    def test_initial_state_is_idle(self):
        """测试初始状态为 IDLE。"""
        sm = AgentStateMachine(AgentState.IDLE)
        assert sm.current_state == AgentState.IDLE

    def test_initial_state_is_running(self):
        """测试可以指定其他初始状态。"""
        sm = AgentStateMachine(AgentState.RUNNING)
        assert sm.current_state == AgentState.RUNNING

    def test_normal_flow_idle_to_completed(self):
        """测试正常流程: IDLE → RUNNING → COMPLETED。"""
        sm = AgentStateMachine(AgentState.IDLE)

        # IDLE → RUNNING
        sm.trigger("start")
        assert sm.current_state == AgentState.RUNNING

        # RUNNING → COMPLETED
        sm.trigger("finish")
        assert sm.current_state == AgentState.COMPLETED

    def test_normal_flow_with_children(self):
        """测试带子任务的流程: IDLE → RUNNING → WAITING_FOR_CHILDREN → RUNNING → COMPLETED。"""
        sm = AgentStateMachine(AgentState.IDLE)

        # IDLE → RUNNING
        sm.trigger("start")
        assert sm.current_state == AgentState.RUNNING

        # RUNNING → WAITING_FOR_CHILDREN
        sm.trigger("spawn_children")
        assert sm.current_state == AgentState.WAITING_FOR_CHILDREN

        # WAITING_FOR_CHILDREN → RUNNING
        sm.trigger("children_settled")
        assert sm.current_state == AgentState.RUNNING

        # RUNNING → COMPLETED
        sm.trigger("finish")
        assert sm.current_state == AgentState.COMPLETED

    def test_loop_flow_multiple_waiting_cycles(self):
        """测试循环流程: 多次 WAITING_FOR_CHILDREN ↔ RUNNING 循环。"""
        sm = AgentStateMachine(AgentState.IDLE)

        # IDLE → RUNNING
        sm.trigger("start")
        assert sm.current_state == AgentState.RUNNING

        # First cycle: RUNNING → WAITING_FOR_CHILDREN → RUNNING
        sm.trigger("spawn_children")
        assert sm.current_state == AgentState.WAITING_FOR_CHILDREN

        sm.trigger("children_settled")
        assert sm.current_state == AgentState.RUNNING

        # Second cycle: RUNNING → WAITING_FOR_CHILDREN → RUNNING
        sm.trigger("spawn_children")
        assert sm.current_state == AgentState.WAITING_FOR_CHILDREN

        sm.trigger("children_settled")
        assert sm.current_state == AgentState.RUNNING

        # Third cycle: RUNNING → WAITING_FOR_CHILDREN → RUNNING
        sm.trigger("spawn_children")
        assert sm.current_state == AgentState.WAITING_FOR_CHILDREN

        sm.trigger("children_settled")
        assert sm.current_state == AgentState.RUNNING

        # Finally: RUNNING → COMPLETED
        sm.trigger("finish")
        assert sm.current_state == AgentState.COMPLETED

    def test_error_flow_from_running(self):
        """测试错误流程: RUNNING → ERROR。"""
        sm = AgentStateMachine(AgentState.RUNNING)

        # RUNNING → ERROR
        sm.trigger("error")
        assert sm.current_state == AgentState.ERROR

    def test_error_flow_full_path(self):
        """测试错误流程完整路径: IDLE → RUNNING → ERROR。"""
        sm = AgentStateMachine(AgentState.IDLE)

        sm.trigger("start")
        assert sm.current_state == AgentState.RUNNING

        sm.trigger("error")
        assert sm.current_state == AgentState.ERROR

    def test_invalid_transition_completed_to_running(self):
        """测试无效转换: COMPLETED → RUNNING 应该抛出异常。"""
        sm = AgentStateMachine(AgentState.COMPLETED)

        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.trigger("start")

        assert exc_info.value.current_state == AgentState.COMPLETED
        assert exc_info.value.event == "start"
        assert "cannot trigger 'start' from state 'completed'" in str(exc_info.value)

    def test_invalid_transition_idle_to_completed(self):
        """测试无效转换: IDLE → COMPLETED 应该抛出异常。"""
        sm = AgentStateMachine(AgentState.IDLE)

        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.trigger("finish")

        assert exc_info.value.current_state == AgentState.IDLE
        assert exc_info.value.event == "finish"

    def test_invalid_transition_waiting_for_children_to_completed(self):
        """测试无效转换: WAITING_FOR_CHILDREN → COMPLETED 应该抛出异常。"""
        sm = AgentStateMachine(AgentState.IDLE)

        # 先进入 WAITING_FOR_CHILDREN 状态
        sm.trigger("start")
        sm.trigger("spawn_children")
        assert sm.current_state == AgentState.WAITING_FOR_CHILDREN

        # 尝试直接完成
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.trigger("finish")

        assert exc_info.value.current_state == AgentState.WAITING_FOR_CHILDREN
        assert exc_info.value.event == "finish"

    def test_invalid_transition_error_to_any(self):
        """测试 ERROR 状态无法进行任何转换。"""
        sm = AgentStateMachine(AgentState.ERROR)

        # ERROR 无法转换到任何状态
        with pytest.raises(InvalidTransitionError):
            sm.trigger("start")

        with pytest.raises(InvalidTransitionError):
            sm.trigger("finish")

        with pytest.raises(InvalidTransitionError):
            sm.trigger("spawn_children")

        with pytest.raises(InvalidTransitionError):
            sm.trigger("children_settled")

        with pytest.raises(InvalidTransitionError):
            sm.trigger("error")

        assert sm.current_state == AgentState.ERROR

    def test_invalid_transition_idle_to_spawn_children(self):
        """测试无效转换: IDLE 不能直接 spawn_children。"""
        sm = AgentStateMachine(AgentState.IDLE)

        with pytest.raises(InvalidTransitionError):
            sm.trigger("spawn_children")

    def test_can_trigger_returns_true_for_valid_events(self):
        """测试 can_trigger() 对有效事件返回 True。"""
        sm = AgentStateMachine(AgentState.IDLE)

        # IDLE 可以 start
        assert sm.can_trigger("start") is True

        # 其他事件不行
        assert sm.can_trigger("finish") is False
        assert sm.can_trigger("spawn_children") is False
        assert sm.can_trigger("children_settled") is False
        assert sm.can_trigger("error") is False

    def test_can_trigger_returns_true_in_running_state(self):
        """测试 can_trigger() 在 RUNNING 状态返回正确结果。"""
        sm = AgentStateMachine(AgentState.RUNNING)

        # RUNNING 可以 finish, spawn_children, error
        assert sm.can_trigger("finish") is True
        assert sm.can_trigger("spawn_children") is True
        assert sm.can_trigger("error") is True

        # 但不能 start 或 children_settled
        assert sm.can_trigger("start") is False
        assert sm.can_trigger("children_settled") is False

    def test_can_trigger_returns_true_in_waiting_state(self):
        """测试 can_trigger() 在 WAITING_FOR_CHILDREN 状态返回正确结果。"""
        sm = AgentStateMachine(AgentState.WAITING_FOR_CHILDREN)

        # WAITING_FOR_CHILDREN 只能 children_settled
        assert sm.can_trigger("children_settled") is True

        # 其他都不行
        assert sm.can_trigger("start") is False
        assert sm.can_trigger("finish") is False
        assert sm.can_trigger("spawn_children") is False
        assert sm.can_trigger("error") is False

    def test_can_trigger_returns_false_for_invalid_events(self):
        """测试 can_trigger() 对无效事件返回 False。"""
        sm = AgentStateMachine(AgentState.COMPLETED)

        # COMPLETED 不能做任何事
        assert sm.can_trigger("start") is False
        assert sm.can_trigger("finish") is False
        assert sm.can_trigger("spawn_children") is False
        assert sm.can_trigger("children_settled") is False
        assert sm.can_trigger("error") is False
        assert sm.can_trigger("nonexistent") is False

    def test_can_trigger_after_transition(self):
        """测试状态转换后 can_trigger() 的行为更新。"""
        sm = AgentStateMachine(AgentState.IDLE)

        # IDLE 状态
        assert sm.can_trigger("start") is True
        assert sm.can_trigger("finish") is False

        # 转换到 RUNNING
        sm.trigger("start")

        # RUNNING 状态
        assert sm.can_trigger("start") is False
        assert sm.can_trigger("finish") is True
        assert sm.can_trigger("spawn_children") is True

    def test_state_preserved_after_invalid_transition(self):
        """测试无效转换后状态保持不变。"""
        sm = AgentStateMachine(AgentState.IDLE)

        assert sm.current_state == AgentState.IDLE

        try:
            sm.trigger("finish")
        except InvalidTransitionError:
            pass

        # 状态应该仍然是 IDLE
        assert sm.current_state == AgentState.IDLE

    def test_complex_workflow_with_error_in_waiting(self):
        """测试复杂工作流: 在 WAITING 后出错。"""
        sm = AgentStateMachine(AgentState.IDLE)

        sm.trigger("start")
        sm.trigger("spawn_children")

        # WAITING_FOR_CHILDREN 不能直接 error，需要先回到 RUNNING
        assert sm.can_trigger("error") is False

        sm.trigger("children_settled")

        # 回到 RUNNING 后可以 error
        assert sm.can_trigger("error") is True
        sm.trigger("error")
        assert sm.current_state == AgentState.ERROR

    def test_invalid_transition_error_message(self):
        """测试错误消息格式正确。"""
        sm = AgentStateMachine(AgentState.IDLE)

        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.trigger("invalid_event")

        error_msg = str(exc_info.value)
        assert "Invalid transition" in error_msg
        assert "cannot trigger 'invalid_event'" in error_msg
        assert "'idle'" in error_msg

    def test_all_states_are_enum_values(self):
        """测试所有状态都是有效的 AgentState 枚举值。"""
        for state in AgentState:
            sm = AgentStateMachine(state)
            assert sm.current_state == state
