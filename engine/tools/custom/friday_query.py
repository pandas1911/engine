"""Friday enterprise database query tool — SQL direct-pass mode.

Allows the AI Agent to query the Friday enterprise database using
raw SQL statements. Only SELECT queries are permitted; the tool
validates input, enforces read-only access, and formats results
as text tables.
"""

import asyncio
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict

from engine.safety import ResultTruncator
from engine.tools.base import Tool


class FridayQueryTool(Tool):
    """Query the Friday enterprise database with SQL statements.

    Accepts SQL queries (SELECT only), executes them against the
    Friday SQLite database in read-only mode, and returns formatted
    text results. Supports JOINs, subqueries, GROUP BY, and
    standard aggregation functions.
    """

    name = "query_data"

    description = """查询 Friday 企业运营数据库。接受 SQL 查询语句，返回格式化的文本结果。

## 数据库表结构

### departments (部门表)
| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | TEXT | 部门名称（唯一）|
| manager | TEXT | 部门经理 |
| headcount_budget | INTEGER | 编制预算 |
| created_at | TEXT | 创建时间 |

### employees (员工表)
| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | TEXT | 姓名 |
| department_id | INTEGER | 外键→departments.id |
| position | TEXT | 职位 |
| level | TEXT | 职级 (P5/P6/P7/P8/P9/M1/M2/M3) |
| salary_grade | TEXT | 薪资等级 (A/B/C/D/E) |
| hire_date | TEXT | 入职日期 |
| status | TEXT | 状态 (active/inactive/on_leave) |

### attendance (考勤表)
| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| employee_id | INTEGER | 外键→employees.id |
| date | TEXT | 日期 |
| status | TEXT | 状态 (normal/late/absent/early_leave/overtime/sick_leave/annual_leave) |
| work_hours | REAL | 工时 |
| overtime_hours | REAL | 加班时数 |

### projects (项目表)
| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| name | TEXT | 项目名称 |
| department_id | INTEGER | 外键→departments.id |
| status | TEXT | 状态 (planning/in_progress/completed/delayed/cancelled) |
| budget | REAL | 预算 |
| spent | REAL | 已花费 |
| progress | INTEGER | 进度 (0-100) |
| start_date | TEXT | 开始日期 |
| end_date | TEXT | 结束日期 |
| description | TEXT | 描述 |

### financial_records (财务记录表)
| 列名 | 类型 | 说明 |
|------|------|------|
| id | INTEGER | 主键 |
| type | TEXT | 类型 (income/expense) |
| department_id | INTEGER | 外键→departments.id |
| amount | REAL | 金额 |
| record_month | TEXT | 记录月份 |
| category | TEXT | 类别 |
| description | TEXT | 描述 |

## 使用说明
- 仅支持 SELECT 查询（只读）
- 支持 JOIN、子查询、GROUP BY、聚合函数（COUNT/SUM/AVG/MAX/MIN）
- 最多返回 100 行结果
- 使用中文进行数据描述

## 示例查询
1. 查询所有部门: SELECT * FROM departments
2. 查询活跃员工及其部门: SELECT e.name, e.position, d.name AS department FROM employees e JOIN departments d ON e.department_id = d.id WHERE e.status = 'active'
3. 统计各部门人数: SELECT d.name AS department, COUNT(*) AS headcount FROM employees e JOIN departments d ON e.department_id = d.id GROUP BY d.name"""

    parameters = {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": (
                    "SQL query to execute against the enterprise database. "
                    "Only SELECT statements are allowed. Supports JOINs, "
                    "subqueries, GROUP BY, and aggregation functions "
                    "(COUNT/SUM/AVG/MAX/MIN). Maximum 100 rows returned."
                ),
            }
        },
        "required": ["sql"],
    }

    DB_PATH = Path("/Users/sys/Desktop/Friday/friday_data.db")
    _MAX_ROWS = 100
    _MAX_CONTENT_LENGTH = 15000

    async def execute(self, arguments: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Execute a SQL query against the Friday database.

        Args:
            arguments: Tool arguments containing 'sql' key.
            context: Execution context (session, agent, task_id, etc.).

        Returns:
            Formatted query results or error message string.
        """
        sql = arguments.get("sql", "")
        if not sql or not isinstance(sql, str) or not sql.strip():
            return "Query error: sql parameter is required and must be a non-empty string"

        sql = sql.strip()

        # Strip line comments for validation purposes only
        sql_for_validation = re.sub(r"--.*$", "", sql, flags=re.MULTILINE).strip()

        # Reject semicolons to prevent multi-statement injection
        if ";" in sql:
            return "Query error: multi-statement queries are not allowed"

        # Reject ATTACH DATABASE to prevent file access
        if "ATTACH" in sql.upper():
            return "Query error: ATTACH DATABASE is not allowed"

        # Only allow SELECT statements
        first_word = sql_for_validation.split()[0].upper() if sql_for_validation.split() else ""
        if first_word != "SELECT":
            return "Query error: only SELECT queries are allowed"

        # Check database file exists
        if not self.DB_PATH.exists():
            return "查询错误: 数据库文件不存在，请先运行数据生成脚本。"

        try:
            result = await asyncio.to_thread(self._execute_query, sql, self.DB_PATH)
            return result
        except Exception as exc:
            return f"Query error: {exc}"

    @staticmethod
    def _execute_query(sql: str, db_path: Path) -> str:
        """Execute a read-only SQL query synchronously.

        Args:
            sql: Validated SQL query string.
            db_path: Path to the SQLite database file.

        Returns:
            Formatted result string.
        """
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute(sql)

            rows = cursor.fetchmany(FridayQueryTool._MAX_ROWS)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []

            if not rows:
                return "查询完成，无匹配数据。"

            # Format as text table with pipe separators
            header = " | ".join(columns)
            separator = "---"
            row_lines = []
            for row in rows:
                formatted_values = [
                    "NULL" if value is None else str(value) for value in row
                ]
                row_lines.append(" | ".join(formatted_values))

            result = (
                f"查询结果 ({len(rows)} 行):\n"
                f"{separator}\n"
                f"{header}\n"
                f"{separator}\n"
                + "\n".join(row_lines)
            )

            return ResultTruncator.truncate(result, FridayQueryTool._MAX_CONTENT_LENGTH)
        finally:
            if conn is not None:
                conn.close()
