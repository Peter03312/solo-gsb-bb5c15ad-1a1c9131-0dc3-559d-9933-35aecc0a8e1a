"""SQLite 连接管理与初始化。

- 读：每次操作开一条短连接（WAL 下读写不互斥）。
- 写：全局锁串行化，``BEGIN IMMEDIATE`` 立即拿写锁；写操作的整个校验+插入在
  同一事务内完成，保证"整体拒绝且不改历史"的原子性。
"""
from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager

from .config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedures (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    number     TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS versions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    procedure_id  INTEGER NOT NULL REFERENCES procedures(id),
    revision      INTEGER NOT NULL,
    body          TEXT NOT NULL,
    valid_start   TEXT NOT NULL,
    valid_end     TEXT NOT NULL,
    supersedes_id INTEGER REFERENCES versions(id),
    published_at  TEXT NOT NULL,
    UNIQUE(procedure_id, revision)
);

-- 撤回事件：只追加，不删除/改写 versions 行。
-- 一个版本至多被撤回一次（UNIQUE）；撤回时刻 withdrawn_at 由服务端记录，不可回填。
-- 双时间维语义：知悉时刻 k 下该版本可见当且仅当
-- published_at <= k 且（未撤回 或 withdrawn_at > k）。
CREATE TABLE IF NOT EXISTS withdrawals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id   INTEGER NOT NULL UNIQUE REFERENCES versions(id),
    withdrawn_at TEXT NOT NULL
);

-- 作业规程包：班组开工前按（作业时刻, 知悉时刻, 一组规程编号）固定。
-- 包一旦形成即不可变：各规程当时适用版本的正文等字段在 package_items 中
-- 完整留快照，之后任何更正/撤回都只作用于 versions/withdrawals，
-- 按包标识取回的内容永远保持固定时原样。
CREATE TABLE IF NOT EXISTS packages (
    id          TEXT PRIMARY KEY,
    work_time   TEXT NOT NULL,
    known_as_of TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    item_count  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS package_items (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    package_id          TEXT NOT NULL REFERENCES packages(id),
    position            INTEGER NOT NULL,
    procedure_id        INTEGER NOT NULL REFERENCES procedures(id),
    number              TEXT NOT NULL,
    revision            INTEGER NOT NULL,
    body                TEXT NOT NULL,
    valid_start         TEXT NOT NULL,
    valid_end           TEXT NOT NULL,
    supersedes_revision INTEGER,
    published_at        TEXT NOT NULL,
    UNIQUE(package_id, position)
);

-- 前置规程引用：每个版本各自携带一份有序的前置规程编号列表。
-- 与版本行同一事务插入、只追加不改写：首次发布省略视为空；更正省略继承
-- 被替代版本（在写入时复制，基版行不可变，故与按引用继承等价）；显式空
-- 列表清空。引用是否生效完全由所属版本的双时间维可见性决定（适用区间、
-- 发布时间、撤回），因此历史查询天然看到当时的前置列表，无需改写历史。
-- 编号在写入时不校验存在性：前置规程可以晚于引用者发布。
CREATE TABLE IF NOT EXISTS prerequisites (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id INTEGER NOT NULL REFERENCES versions(id),
    position   INTEGER NOT NULL,
    number     TEXT NOT NULL,
    UNIQUE(version_id, position)
);

-- 值班审计事件日志：只追加，供 GET /api/v1/audit/events 按提交顺序增量取回。
-- 一次写事务（首次发布 / 批量更正 / 批量撤回）= 一个 audit_commits 行：
-- 游标只随已提交事务前进，未提交事务的行在任何读连接中均不可见。
-- 同一提交批次的事件共享 commit_seq、按请求顺序 position 排列，分页保证
-- 整批取出、绝不跨页拆开。audit_events.seq 为全站全局提交顺序（AUTOINCREMENT
-- 单调，只增不删），即对外游标依据。事件表不建外键：它是对 versions/
-- withdrawals 的只追加旁录，且升级前的历史要在不改写原记录的前提下补录。
CREATE TABLE IF NOT EXISTS audit_commits (
    commit_seq   INTEGER PRIMARY KEY AUTOINCREMENT,
    committed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    commit_seq INTEGER NOT NULL,
    position   INTEGER NOT NULL,
    op         TEXT NOT NULL,
    number     TEXT NOT NULL,
    revision   INTEGER NOT NULL,
    event_at   TEXT NOT NULL,
    UNIQUE(commit_seq, position)
);

CREATE INDEX IF NOT EXISTS idx_versions_procedure
    ON versions(procedure_id);
CREATE INDEX IF NOT EXISTS idx_versions_query
    ON versions(published_at, valid_start, valid_end);
CREATE INDEX IF NOT EXISTS idx_withdrawals_version
    ON withdrawals(version_id);
CREATE INDEX IF NOT EXISTS idx_package_items_package
    ON package_items(package_id);
CREATE INDEX IF NOT EXISTS idx_prerequisites_version
    ON prerequisites(version_id);
CREATE INDEX IF NOT EXISTS idx_audit_events_commit
    ON audit_events(commit_seq, seq);
"""

_write_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(
        settings.db_path, timeout=settings.busy_timeout_ms / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {int(settings.busy_timeout_ms)}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """幂等建表并补录升级前历史；内存库可安全重复调用。

    建表与历史补录在同一条事务内完成：旧库升级时把升级前已存在的
    versions/withdrawals 行按提交顺序回填进审计日志（每条记录一个提交、
    呈现为单条事件），全程只读旧表、不改写任何原记录。
    """
    if settings.db_path != ":memory:":
        parent = os.path.dirname(os.path.abspath(settings.db_path))
        os.makedirs(parent, exist_ok=True)
    conn = _connect()
    try:
        if settings.db_path != ":memory:":
            conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)
        backfill_audit_log(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# 升级前已落库的版本/撤回行按"服务端时刻"（published_at / withdrawn_at）
# 排序补录；同一时刻的并列按 op 与编号/版本号给出确定次序。首版
# (revision=0) 是首次发布，其余版本行是更正；撤回行一律为 withdrawal。
# 每条旧记录形成一个独立的审计提交（commit 与 event 一一对应、各只有一条
# 事件），因此补录部分同样满足"一页至少装得下一个完整批次"。
_BACKFILL_SQL = """
SELECT v.published_at AS event_at,
       CASE WHEN v.revision = 0 THEN 'release' ELSE 'correction' END AS op,
       p.number AS number, v.revision AS revision
  FROM versions v
  JOIN procedures p ON p.id = v.procedure_id
UNION ALL
SELECT w.withdrawn_at AS event_at, 'withdrawal' AS op,
       p.number AS number, v.revision AS revision
  FROM withdrawals w
  JOIN versions v ON v.id = w.version_id
  JOIN procedures p ON p.id = v.procedure_id
ORDER BY event_at, op, number, revision
"""


def backfill_audit_log(conn: sqlite3.Connection) -> int:
    """把升级前的 versions/withdrawals 行补录进审计日志（幂等）。

    仅在审计事件表为空、而业务表已有记录时执行；补录只读旧表、绝不
    UPDATE/DELETE 原记录。返回补录的事件数。调用方负责提交/回滚。
    """
    existing = conn.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
    if existing:
        return 0
    rows = conn.execute(_BACKFILL_SQL).fetchall()
    for event_at, op, number, revision in rows:
        cur = conn.execute(
            "INSERT INTO audit_commits (committed_at) VALUES (?)",
            (event_at,))
        commit_seq = cur.lastrowid
        conn.execute(
            "INSERT INTO audit_events (commit_seq, position, op, number, "
            "revision, event_at) VALUES (?, 0, ?, ?, ?, ?)",
            (commit_seq, op, number, revision, event_at))
    return len(rows)


@contextmanager
def read_conn() -> Iterator[sqlite3.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def read_snapshot() -> Iterator[sqlite3.Connection]:
    """只读一致性快照：显式开启一条 deferred 事务，并**立即执行一次读**把
    SQLite 的快照固定在进入时刻（deferred 事务的快照在首次读时取得，而非
    BEGIN 时）。此后事务内所有读取看到的都是同一数据库版本，不会拼接并发
    写入前后的结果（WAL 下读事务在该快照上保持到事务结束）。
    """
    conn = _connect()
    try:
        conn.execute("BEGIN")
        # 立即读真实表以获取 WAL 读锁、把读快照固定在进入时刻（SELECT 1 不
        # 访问任何表，不取读锁也不建立快照；deferred 事务的快照在首次表读时
        # 取得）。此后事务内所有读取都基于同一数据库版本。
        conn.execute("SELECT 1 FROM packages LIMIT 1").fetchone()
        yield conn
        conn.rollback()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def write_tx() -> Iterator[sqlite3.Connection]:
    """串行化的写事务：异常即回滚，绝不留下半截历史。"""
    with _write_lock:
        conn = _connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
