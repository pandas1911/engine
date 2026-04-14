"""Custom tools for the Agent system."""

from engine.tools.base import Tool


class MockTool(Tool):
    name = "get_weather"
    description = "获取天气的工具，返回天气信息"
    parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "要查询天气的位置"},
        },
        "required": ["location"],
    }

    async def execute(self, arguments: dict, context: dict) -> str:
        location = arguments.get("location", "")
        return str(self._get_value(location))

    def _get_value(self, location: str) -> int:
        if "上海" in location or "shanghai" in location.lower():
            return "服务器错误，请稍后重试"
        elif "北京" in location or "beijing" in location.lower():
            return 24
        return 22
