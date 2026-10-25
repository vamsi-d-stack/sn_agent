"""Run with: python -m streamlit run streamlit_app.py.

Set OPENROUTER_API_KEY (or OPENAI_API_KEY) and the ServiceNow environment
variables before launching. OPENROUTER_MODEL and SN_MCP_SERVER are optional.
"""

import asyncio
import json
import os

import streamlit as st

from deploy_test_snagent.openai_sn_mcp import chat_connection, initial_messages, list_chat_tools


async def advance(messages: list[dict], approved: bool | None = None) -> None:
    async with chat_connection() as (session, client):
        tools = await list_chat_tools(session)
        if approved is not None:
            calls = messages[-1]["tool_calls"]
            for call in calls:
                if not approved:
                    output = json.dumps({"error": "Tool execution denied by user"})
                else:
                    try:
                        arguments = json.loads(call["function"]["arguments"] or "{}")
                        if not isinstance(arguments, dict):
                            raise ValueError("Tool arguments must be a JSON object")
                        result = await session.call_tool(
                            call["function"]["name"], arguments=arguments
                        )
                        output = result.model_dump_json(exclude_none=True)
                    except Exception as exc:
                        st.error(f"Tool {call['function']['name']} failed: {exc}")
                        output = json.dumps({"error": str(exc)})
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": output}
                )

        response = await client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            messages=messages,
            **({"tools": tools, "tool_choice": "auto"} if tools else {}),
        )
        message = response.choices[0].message
        entry = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            entry["tool_calls"] = [
                call.model_dump(exclude_none=True) for call in message.tool_calls
            ]
        messages.append(entry)


def main() -> None:
    st.set_page_config(page_title="ServiceNow Assistant")
    st.title("ServiceNow Assistant")
    st.caption("Ask about incidents, changes, and tasks. Tool calls require approval.")

    if "messages" not in st.session_state:
        st.session_state.messages = initial_messages()
        st.session_state.rounds = 0
        st.session_state.failed = False
    if st.sidebar.button("Clear chat"):
        st.session_state.messages = initial_messages()
        st.session_state.rounds = 0
        st.session_state.failed = False
        st.rerun()

    messages = st.session_state.messages
    for message in messages:
        if message["role"] in {"user", "assistant"}:
            with st.chat_message(message["role"]):
                if message.get("content"):
                    st.markdown(message["content"])
                for call in message.get("tool_calls", []):
                    with st.expander(f"Tool: {call['function']['name']}"):
                        st.code(call["function"]["arguments"], language="json")
        elif message["role"] == "tool":
            with st.expander("Tool result"):
                st.code(message["content"], language="json")

    pending = bool(messages[-1].get("tool_calls"))
    approved = None
    action = False
    if st.session_state.failed:
        st.error("The request failed. Clear chat to start a new conversation.")
    elif pending:
        st.info("Review the tool calls above before allowing execution.")
        allow, deny = st.columns(2)
        if allow.button("Approve listed tools", type="primary"):
            approved, action = True, True
        if deny.button("Deny listed tools"):
            approved, action = False, True

    prompt = st.chat_input(
        "Enter your prompt",
        disabled=pending or st.session_state.failed,
    )
    if prompt and prompt.strip():
        messages.append({"role": "user", "content": prompt.strip()})
        st.session_state.rounds = 0
        action = True

    if action:
        try:
            with st.spinner("Working..."):
                asyncio.run(advance(messages, approved))
            st.session_state.rounds += 1
            if messages[-1].get("tool_calls") and st.session_state.rounds >= 15:
                st.session_state.failed = True
                st.error("Tool-call limit reached. Clear chat to continue.")
                st.stop()
        except Exception as exc:
            st.session_state.failed = True
            st.error(f"Request failed: {exc}")
            st.exception(exc)
            st.stop()
        st.rerun()


if __name__ == "__main__":
    main()
