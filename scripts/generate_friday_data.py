"""
Generate a SQLite database with realistic Chinese enterprise simulation data.

Creates 5 tables: departments, employees, attendance, projects, financial_records.
Idempotent: deletes any existing database before recreating.
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("/Users/sys/Desktop/Friday/friday_data.db")

# Seed for reproducibility (remove for different data each run)
random.seed(42)

# ── Reference data ──────────────────────────────────────────────────────────

SURNAMES = [
    "张", "李", "王", "刘", "陈", "杨", "赵", "黄", "周", "吴",
    "徐", "孙", "胡", "朱", "高", "林", "何", "郭", "马", "罗",
    "梁", "宋", "郑", "谢", "韩", "唐", "冯", "于", "董", "萧",
]

GIVEN_NAMES = [
    "伟", "芳", "娜", "秀英", "敏", "静", "强", "磊", "洋", "勇",
    "军", "杰", "娟", "艳", "涛", "明", "超", "秀兰", "霞", "平",
    "刚", "桂英", "文", "华", "飞", "玉兰", "慧", "鑫", "志强", "建华",
    "建国", "建军", "婷", "雪", "浩", "宇", "思远", "晓峰", "佳", "颖",
    "旭", "鹏", "磊", "鑫", "梦", "雨", "思琪", "文博", "子轩", "嘉豪",
]

DEPARTMENT_DEFS = [
    {"name": "研发部", "manager": "李强", "headcount_budget": 20},
    {"name": "市场部", "manager": "王芳", "headcount_budget": 15},
    {"name": "销售部", "manager": "陈伟", "headcount_budget": 18},
    {"name": "人力资源部", "manager": "张敏", "headcount_budget": 10},
    {"name": "财务部", "manager": "刘静", "headcount_budget": 8},
]

POSITIONS_BY_DEPT = {
    "研发部": [
        "高级工程师", "前端工程师", "后端工程师", "测试工程师",
        "架构师", "技术经理", "DevOps工程师", "算法工程师",
        "数据工程师", "移动开发工程师",
    ],
    "市场部": [
        "市场经理", "品牌专员", "内容运营", "活动策划",
        "市场分析师", "数字营销专员", "公关经理", "新媒体运营",
    ],
    "销售部": [
        "销售经理", "客户经理", "销售代表", "大客户经理",
        "渠道经理", "售前顾问", "商务拓展", "区域经理",
    ],
    "人力资源部": [
        "HR经理", "招聘专员", "薪酬专员", "培训专员",
        "员工关系专员", "HRBP", "组织发展专员",
    ],
    "财务部": [
        "财务经理", "会计", "出纳", "审计专员",
        "税务专员", "成本会计", "财务分析师",
    ],
}

LEVELS = ["P5", "P6", "P7", "P8", "P9", "M1", "M2", "M3"]
SALARY_GRADES = ["A", "B", "C", "D", "E"]

# Level → typical salary grade mapping (higher level → higher grade)
LEVEL_GRADE_MAP = {
    "P5": ["D", "E"],
    "P6": ["C", "D", "E"],
    "P7": ["C", "D"],
    "P8": ["B", "C"],
    "P9": ["A", "B"],
    "M1": ["B", "C"],
    "M2": ["A", "B"],
    "M3": ["A"],
}

ATTENDANCE_STATUSES = ["normal", "late", "absent", "early_leave", "overtime", "sick_leave", "annual_leave"]
# Weights: normal ~75%, late ~10%, absent ~5%, early_leave ~3%, overtime ~4%, sick_leave ~2%, annual_leave ~1%
ATTENDANCE_WEIGHTS = [75, 10, 5, 3, 4, 2, 1]

PROJECT_NAMES = [
    ("智能客服系统", "研发部"),
    ("数据中台建设", "研发部"),
    ("移动端App改版", "研发部"),
    ("微服务架构迁移", "研发部"),
    ("品牌升级项目", "市场部"),
    ("数字营销平台", "市场部"),
    ("年度营销计划", "市场部"),
    ("大客户拓展计划", "销售部"),
    ("新零售渠道建设", "销售部"),
    ("区域市场开拓", "销售部"),
    ("销售系统升级", "销售部"),
    ("人才培养计划", "人力资源部"),
    ("绩效管理体系优化", "人力资源部"),
    ("年度预算编制", "财务部"),
    ("财务系统升级", "财务部"),
    ("成本优化项目", "财务部"),
    ("AI辅助决策平台", "研发部"),
    ("社交媒体推广", "市场部"),
    ("企业文化建设", "人力资源部"),
    ("合规审计整改", "财务部"),
]

PROJECT_STATUSES = ["planning", "in_progress", "completed", "delayed", "cancelled"]

FINANCIAL_CATEGORIES = ["工资", "设备", "营销", "差旅", "办公", "培训", "软件", "咨询", "维护", "其他"]

# ── Helper functions ────────────────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> str:
    """Return a random date string between start and end inclusive."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def random_chinese_name() -> str:
    """Generate a random Chinese name from predefined pools."""
    return random.choice(SURNAMES) + random.choice(GIVEN_NAMES)


def generate_unique_names(count: int) -> list[str]:
    """Generate `count` unique Chinese names."""
    names: set[str] = set()
    while len(names) < count:
        names.add(random_chinese_name())
    return list(names)


def workdays_between(start: datetime, end: datetime) -> list[str]:
    """Return a list of weekday date strings between start and end."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0 .. Friday=4
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


# ── Table creation ──────────────────────────────────────────────────────────

def create_tables(cursor: sqlite3.Cursor) -> None:
    """Create all 5 tables with exact schemas."""
    cursor.execute("""
        CREATE TABLE departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            manager TEXT NOT NULL,
            headcount_budget INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)

    cursor.execute("""
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            position TEXT NOT NULL,
            level TEXT NOT NULL CHECK(level IN ('P5','P6','P7','P8','P9','M1','M2','M3')),
            hire_date TEXT NOT NULL,
            salary_grade TEXT NOT NULL CHECK(salary_grade IN ('A','B','C','D','E')),
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','inactive','on_leave')),
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
    """)

    cursor.execute("""
        CREATE TABLE attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('normal','late','absent','early_leave','overtime','sick_leave','annual_leave')),
            work_hours REAL NOT NULL DEFAULT 8.0,
            overtime_hours REAL NOT NULL DEFAULT 0.0,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        );
    """)

    cursor.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('planning','in_progress','completed','delayed','cancelled')),
            budget REAL NOT NULL,
            spent REAL NOT NULL DEFAULT 0.0,
            progress INTEGER NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 100),
            start_date TEXT NOT NULL,
            end_date TEXT,
            description TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
    """)

    cursor.execute("""
        CREATE TABLE financial_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL CHECK(type IN ('income','expense')),
            department_id INTEGER NOT NULL,
            amount REAL NOT NULL,
            record_month TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        );
    """)


# ── Data insertion ──────────────────────────────────────────────────────────

def insert_departments(cursor: sqlite3.Cursor) -> dict[str, int]:
    """Insert 5 predefined departments. Returns {name: id} mapping."""
    dept_ids: dict[str, int] = {}
    for dept in DEPARTMENT_DEFS:
        cursor.execute(
            "INSERT INTO departments (name, manager, headcount_budget) VALUES (?, ?, ?)",
            (dept["name"], dept["manager"], dept["headcount_budget"]),
        )
        dept_ids[dept["name"]] = cursor.lastrowid  # type: ignore[assignment]
    print(f"  departments: {len(dept_ids)} rows")
    return dept_ids


def insert_employees(cursor: sqlite3.Cursor, dept_ids: dict[str, int]) -> list[int]:
    """Insert 60-80 employees across departments with realistic levels/grades."""
    # Target ~70 employees, distributed by department proportionally to budget
    total_budget = sum(d["headcount_budget"] for d in DEPARTMENT_DEFS)
    target_total = random.randint(60, 80)

    # Managers first (M1-M3 levels, one per department)
    active_ids: list[int] = []

    for dept_def in DEPARTMENT_DEFS:
        dept_name = dept_def["name"]
        dept_id = dept_ids[dept_name]

        # Insert the manager with a management level
        manager_level = random.choice(["M1", "M2", "M3"])
        manager_grade = random.choice(LEVEL_GRADE_MAP[manager_level])
        manager_name = dept_def["manager"]
        hire = random_date(datetime(2020, 1, 1), datetime(2023, 6, 30))

        cursor.execute(
            "INSERT INTO employees (name, department_id, position, level, hire_date, salary_grade, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (manager_name, dept_id, dept_def["name"].replace("部", "经理"), manager_level, hire, manager_grade, "active"),
        )
        active_ids.append(cursor.lastrowid)  # type: ignore[arg-type]

    # Distribute remaining employees proportionally
    managers_count = len(DEPARTMENT_DEFS)
    remaining = target_total - managers_count

    for dept_def in DEPARTMENT_DEFS:
        dept_name = dept_def["name"]
        dept_id = dept_ids[dept_name]
        proportion = dept_def["headcount_budget"] / total_budget
        dept_count = max(3, round(remaining * proportion))

        positions = POSITIONS_BY_DEPT[dept_name]
        names = generate_unique_names(dept_count)

        for i in range(dept_count):
            # Weight towards mid-levels (P6, P7) with fewer P5 and P9
            level = random.choices(
                ["P5", "P6", "P6", "P7", "P7", "P8", "P9"],
                weights=[15, 25, 20, 20, 10, 7, 3],
            )[0]
            grade = random.choice(LEVEL_GRADE_MAP[level])
            position = random.choice(positions)
            hire = random_date(datetime(2020, 1, 1), datetime(2025, 12, 31))

            # ~80% active, ~15% inactive, ~5% on_leave
            status = random.choices(
                ["active", "inactive", "on_leave"],
                weights=[80, 15, 5],
            )[0]

            cursor.execute(
                "INSERT INTO employees (name, department_id, position, level, hire_date, salary_grade, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (names[i], dept_id, position, level, hire, grade, status),
            )
            if status == "active":
                active_ids.append(cursor.lastrowid)  # type: ignore[arg-type]

    total = cursor.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
    print(f"  employees: {total} rows")
    return active_ids


def insert_attendance(cursor: sqlite3.Cursor, active_ids: list[int]) -> None:
    """Insert ~200 attendance records for recent 3 months of workdays."""
    today = datetime.now()
    three_months_ago = today - timedelta(days=90)
    workdays = workdays_between(three_months_ago, today)

    # Pick a subset of workdays and a subset of active employees to get ~200 records
    # Strategy: for ~70% of workdays, pick 1-2 random active employees
    target = random.randint(200, 210)
    count = 0
    n_active = len(active_ids)

    records = []
    for day in workdays:
        # Each workday, pick 2-4 random active employees
        n_pick = random.randint(2, 4)
        selected = random.sample(active_ids, min(n_pick, n_active))
        for emp_id in selected:
            status = random.choices(ATTENDANCE_STATUSES, weights=ATTENDANCE_WEIGHTS)[0]

            work_hours = 8.0
            overtime_hours = 0.0

            if status == "normal":
                work_hours = round(random.uniform(7.5, 8.5), 1)
            elif status == "late":
                work_hours = round(random.uniform(6.0, 7.5), 1)
            elif status == "absent":
                work_hours = 0.0
            elif status == "early_leave":
                work_hours = round(random.uniform(4.0, 6.5), 1)
            elif status == "overtime":
                work_hours = 8.0
                overtime_hours = round(random.uniform(1.0, 4.0), 1)
            elif status in ("sick_leave", "annual_leave"):
                work_hours = 0.0

            records.append((emp_id, day, status, work_hours, overtime_hours))
            count += 1

    # Trim to target if we overshot
    if count > target:
        random.shuffle(records)
        records = records[:target]
        count = len(records)

    for rec in records:
        cursor.execute(
            "INSERT INTO attendance (employee_id, date, status, work_hours, overtime_hours) VALUES (?, ?, ?, ?, ?)",
            rec,
        )

    print(f"  attendance: {count} rows")


def insert_projects(cursor: sqlite3.Cursor, dept_ids: dict[str, int]) -> None:
    """Insert 15-20 projects across departments with mixed statuses."""
    # Use first 15-20 from PROJECT_NAMES
    n_projects = random.randint(15, 20)
    projects = PROJECT_NAMES[:n_projects]

    for proj_name, dept_name in projects:
        dept_id = dept_ids[dept_name]

        status = random.choices(
            PROJECT_STATUSES,
            weights=[10, 25, 30, 20, 15],
        )[0]

        # Budget in range 50k-2M CNY
        budget = round(random.uniform(50_000, 2_000_000), 2)

        # Start date in 2024-2025 range
        start = random_date(datetime(2024, 1, 1), datetime(2025, 6, 30))

        # Progress and spent depend on status
        if status == "planning":
            progress = random.randint(0, 15)
            spent = round(budget * random.uniform(0, 0.1), 2)
            end_date = None
        elif status == "in_progress":
            progress = random.randint(20, 85)
            spent = round(budget * (progress / 100) * random.uniform(0.8, 1.1), 2)
            end_date = None
        elif status == "completed":
            progress = 100
            spent = round(budget * random.uniform(0.85, 1.05), 2)
            end_date = random_date(
                datetime.strptime(start, "%Y-%m-%d") + timedelta(days=30),
                datetime(2026, 3, 31),
            )
        elif status == "delayed":
            progress = random.randint(30, 75)
            # Delayed projects may exceed budget by up to 20%
            spent = round(budget * min(1.2, (progress / 100) * random.uniform(0.9, 1.3)), 2)
            end_date = None
        else:  # cancelled
            progress = random.randint(5, 40)
            spent = round(budget * (progress / 100) * random.uniform(0.5, 1.0), 2)
            end_date = None

        description = f"{dept_name}{proj_name}项目"

        cursor.execute(
            "INSERT INTO projects (name, department_id, status, budget, spent, progress, start_date, end_date, description) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (proj_name, dept_id, status, budget, spent, progress, start, end_date, description),
        )

    count = cursor.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    print(f"  projects: {count} rows")


def insert_financial_records(cursor: sqlite3.Cursor, dept_ids: dict[str, int]) -> None:
    """Insert 100-120 financial records for recent 6 months."""
    today = datetime.now()
    six_months_ago = today - timedelta(days=180)

    # Generate month strings for the last 6 months
    months: list[str] = []
    current = six_months_ago.replace(day=1)
    while current <= today:
        months.append(current.strftime("%Y-%m"))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    target = random.randint(100, 120)
    count = 0
    dept_list = list(dept_ids.values())

    # Category → typical type mapping
    expense_categories = ["工资", "设备", "营销", "差旅", "办公", "培训", "软件", "咨询", "维护", "其他"]
    income_categories = ["营销", "软件", "咨询", "其他"]

    descriptions_map = {
        "工资": ["月度工资发放", "绩效奖金", "加班费", "年终奖"],
        "设备": ["服务器采购", "办公电脑采购", "网络设备升级", "实验室设备"],
        "营销": ["广告投放", "展会费用", "市场推广", "品牌合作"],
        "差旅": ["出差补贴", "客户拜访", "会议差旅", "培训差旅"],
        "办公": ["办公用品采购", "会议室租赁", "物业水电", "快递打印"],
        "培训": ["内部培训", "外部培训", "认证考试", "培训资料"],
        "软件": ["软件许可", "云服务订阅", "开发工具", "安全软件"],
        "咨询": ["管理咨询", "技术咨询", "法律顾问", "审计服务"],
        "维护": ["系统维护", "设备维修", "安全维护", "基础设施维护"],
        "其他": ["杂项支出", "团建活动", "福利费用", "其他"],
    }

    while count < target:
        dept_id = random.choice(dept_list)
        month = random.choice(months)

        # ~35% income, ~65% expense
        if random.random() < 0.35:
            record_type = "income"
            category = random.choice(income_categories)
            amount = round(random.uniform(50_000, 800_000), 2)
        else:
            record_type = "expense"
            category = random.choice(expense_categories)
            # Expenses vary by category
            if category == "工资":
                amount = round(random.uniform(100_000, 500_000), 2)
            elif category == "设备":
                amount = round(random.uniform(10_000, 300_000), 2)
            else:
                amount = round(random.uniform(1_000, 100_000), 2)

        description = random.choice(descriptions_map.get(category, ["其他"]))

        cursor.execute(
            "INSERT INTO financial_records (type, department_id, amount, record_month, category, description) VALUES (?, ?, ?, ?, ?, ?)",
            (record_type, dept_id, amount, month, category, description),
        )
        count += 1

    print(f"  financial_records: {count} rows")


# ── Verification ────────────────────────────────────────────────────────────

def verify_data(cursor: sqlite3.Cursor) -> None:
    """Print row counts and key distributions for verification."""
    print("\n── Verification ─────────────────────────────────")

    tables = ["departments", "employees", "attendance", "projects", "financial_records"]
    for table in tables:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    print("\n  Employee status distribution:")
    for row in cursor.execute("SELECT status, COUNT(*) FROM employees GROUP BY status"):
        print(f"    {row[0]}: {row[1]}")

    print("\n  Project status distribution:")
    for row in cursor.execute("SELECT status, COUNT(*) FROM projects GROUP BY status"):
        print(f"    {row[0]}: {row[1]}")

    print("\n  Attendance status distribution:")
    for row in cursor.execute("SELECT status, COUNT(*) FROM attendance GROUP BY status"):
        print(f"    {row[0]}: {row[1]}")

    print("\n  Financial records type distribution:")
    for row in cursor.execute("SELECT type, COUNT(*) FROM financial_records GROUP BY type"):
        print(f"    {row[0]}: {row[1]}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    # Idempotent: delete existing DB
    if DB_PATH.exists():
        DB_PATH.unlink()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    print("Generating Friday database...")
    create_tables(cursor)
    dept_ids = insert_departments(cursor)
    active_ids = insert_employees(cursor, dept_ids)
    insert_attendance(cursor, active_ids)
    insert_projects(cursor, dept_ids)
    insert_financial_records(cursor, dept_ids)

    conn.commit()
    verify_data(cursor)
    conn.close()
    print(f"\nDatabase generated: {DB_PATH}")


if __name__ == "__main__":
    main()
