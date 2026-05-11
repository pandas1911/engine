"""Environment context builder for system prompt injection.

Collects context from multiple sources (time, config, system)
into a flat dict for the builder's ## Environment layer.
"""

import platform
from datetime import datetime
from typing import Dict, Optional
from zoneinfo import ZoneInfo


def build_env_context(
    time_provider,                    # TimeProvider instance
    workspace_dir: str,               # resolved from Config.get_workspace_path()
    model_name: str,                  # primary model label
    platform_override: Optional[str] = None,  # for testing
) -> Dict[str, str]:
    """Collect environment context from multiple sources.

    Returns a flat dict suitable for builder.py's ## Environment layer.
    """
    tz_name = time_provider.resolve_timezone()
    tz_obj = ZoneInfo(tz_name)
    date_str = datetime.now(tz=tz_obj).strftime("%a %b %d %Y")

    return {
        "Date": date_str,                   # e.g. "Tue May 05 2026"
        "Timezone": tz_name,                # e.g. "Asia/Shanghai"
        "Working Directory": workspace_dir,  # e.g. "/Users/sys/Desktop/Friday"
        "Model": model_name,                # e.g. "gpt-4o"
        "OS": platform_override or platform.system(),  # e.g. "Darwin"
    }
