"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    thread_id = st.session_state.get("thread_id", "current")
    feedback = st.text_input("Feedback (optional)", key=f"approval_feedback_{thread_id}")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.subheader(f"Strong escalation — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    thread_id = st.session_state.get("thread_id", "current")
    with st.form(f"escalation_{thread_id}"):
        answers = {
            question: st.text_input(question, key=f"escalation_answer_{thread_id}_{index}")
            for index, question in enumerate(payload["questions"])
        }
        if st.form_submit_button("Submit answers"):
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


async def recent_sessions() -> list[dict]:
    """Reuse replay's grouped session query, ranking risk by severity."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   CASE MAX(CASE risk_level
                       WHEN 'high' THEN 3 WHEN 'med' THEN 2 WHEN 'low' THEN 1 ELSE 0 END)
                       WHEN 3 THEN 'high' WHEN 2 THEN 'med' WHEN 1 THEN 'low'
                       ELSE 'unknown' END AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 25
            """
        ) as cur:
            rows = await cur.fetchall()
    return [dict(row) for row in rows]


async def load_session(thread_id: str) -> dict:
    """Read a saved checkpoint without invoking graph nodes or posting comments."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
    interrupts = [item for task in snapshot.tasks for item in task.interrupts]
    return {
        "values": dict(snapshot.values),
        "interrupt_payload": interrupts[0].value if interrupts else None,
    }


def _store_result(result: dict) -> None:
    interrupts = result.get("__interrupt__", ())
    st.session_state.interrupt_payload = interrupts[0].value if interrupts else None
    st.session_state.final = None if interrupts else result
    st.session_state.error = None


def _error_message(exc: Exception) -> str:
    """Report status without exposing credentials or response bodies."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 404 and "api.github.com" in str(getattr(response, "url", "")):
        return "BLOCKED BY GITHUB PR-Demo 404 — repository access not granted."
    if status is not None:
        return f"Review could not complete (HTTP {status}). Check access and configuration."
    return f"Review could not complete ({type(exc).__name__}). Check configuration and saved session."


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Recent sessions")
        try:
            sessions = asyncio.run(recent_sessions())
        except Exception as exc:
            st.error(f"Could not load audit sessions ({type(exc).__name__}).")
            return
        if not sessions:
            st.caption("No audit sessions yet.")
            return
        st.dataframe(
            [{key: row[key] for key in ("thread_id", "pr_url", "worst_risk", "last_event")}
             for row in sessions],
            hide_index=True,
        )
        selected = st.selectbox("Session", options=[row["thread_id"] for row in sessions])
        if st.button("Open session"):
            selected_session = next(row for row in sessions if row["thread_id"] == selected)
            try:
                saved = asyncio.run(load_session(selected))
            except Exception as exc:
                st.error(f"Could not read checkpoint ({type(exc).__name__}).")
                return
            values = saved["values"]
            st.session_state.thread_id = selected
            st.session_state.pr_url = values.get("pr_url", selected_session["pr_url"])
            st.session_state.interrupt_payload = saved["interrupt_payload"]
            st.session_state.final = values if values.get("final_action") else None
            st.session_state.error = None
            if not values:
                st.session_state.error = "Audit events exist, but no saved graph checkpoint was found."
            elif not values.get("final_action") and saved["interrupt_payload"] is None:
                st.session_state.error = "This session has no completed review or pending human request."
            st.rerun()


def _render_final(final: dict) -> None:
    action = final.get("final_action") or "unknown"
    # auto_commit_failed has the auto prefix too; check failures before success.
    if action.endswith("failed"):
        st.error(f"{action} — the GitHub comment was not confirmed as posted.")
    elif action.startswith("auto") or action.startswith("committed"):
        st.success(f"Review completed — {action}")
    elif action == "rejected":
        st.warning("Rejected / skipped — no comment posted.")
    else:
        st.info(f"final_action = {action}")
    st.write(f"Final action: {action}")
    analysis = final.get("analysis")
    if analysis is not None:
        data = analysis.model_dump() if hasattr(analysis, "model_dump") else analysis
        st.metric("Confidence", f"{data['confidence']:.0%}")
        st.markdown(data["summary"])
    pr_url = final.get("pr_url", st.session_state.pr_url)
    st.markdown(f"[View PR on GitHub]({pr_url})")
    thread_id = st.session_state.thread_id
    st.caption(f"thread_id = {thread_id}")
    st.code(f"uv run python -m audit.replay --thread {thread_id}", language="bash")


# ─── Main flow ─────────────────────────────────────────────────────────────
def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="HITL PR Review", layout="wide")
    for key, default in {
        "thread_id": None, "pr_url": "", "interrupt_payload": None,
        "final": None, "error": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    st.title("HITL PR Review Agent")
    _render_sidebar()

    with st.form("start"):
        pr_url = st.text_input(
            "PR URL", value=st.session_state.pr_url,
            placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
        )
        submitted = st.form_submit_button("Run review")

    if submitted and pr_url:
        st.session_state.pr_url = pr_url
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.interrupt_payload = None
        st.session_state.final = None
        st.session_state.error = None
        try:
            with st.spinner("Fetching PR + asking the LLM..."):
                result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))
            _store_result(result)
        except Exception as exc:
            st.session_state.error = _error_message(exc)

    payload = st.session_state.interrupt_payload
    if payload is not None:
        kind = payload["kind"]
        if kind == "approval_request":
            answer = render_approval_card(payload)
        elif kind == "escalation":
            answer = render_escalation_card(payload)
        else:
            st.error(f"Unsupported interrupt kind: {kind}")
            answer = None
        if answer is not None:
            # Consume the UI request before resuming. A passive rerun must never
            # retry a failed commit or replay the user's previous submission.
            st.session_state.interrupt_payload = None
            try:
                with st.spinner("Resuming..."):
                    result = asyncio.run(run_graph(
                        st.session_state.pr_url, st.session_state.thread_id,
                        resume_value=answer,
                    ))
                _store_result(result)
            except Exception as exc:
                st.session_state.error = _error_message(exc)
            st.rerun()

    if st.session_state.error:
        st.error(st.session_state.error)
    if st.session_state.final is not None:
        _render_final(st.session_state.final)


if __name__ == "__main__":
    main()
