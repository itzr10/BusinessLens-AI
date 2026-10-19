"""BusinessLens AI: Streamlit dashboard.

Run with:  python -m streamlit run app.py

This file only arranges the page. Every number comes from the tested functions
in ``businesslens/analytics.py``; filtering/formatting helpers live in
``businesslens/dashboard_helpers.py``.
"""

from pathlib import Path

import plotly.express as px
import streamlit as st

from businesslens import agent as ai
from businesslens import analytics as an
from businesslens import dashboard_helpers as dh
from businesslens.providers import PROVIDERS, ProviderConfig, ProviderError, create_provider

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_sales.csv"
BAR_COLOR = "#2f6fdb"
PROFIT_COLOR = "#1a9f6e"

st.set_page_config(page_title="BusinessLens AI", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------
# Loading data (everything stays in memory; uploads are never saved to disk)
# --------------------------------------------------------------------------

def load_data():
    """Return (validated dataframe, source label), or (None, label) after showing an error."""
    uploaded = st.sidebar.file_uploader(
        "Upload your own sales CSV",
        type=["csv"],
        help="Required columns: order_date, product, region, revenue, cost.",
    )
    if uploaded is None:
        label = "Sample data (data/sample_sales.csv)"
        try:
            return an.load_sales_data(SAMPLE_PATH), label
        except an.DataValidationError as exc:
            st.error(f"The sample dataset could not be loaded: {exc}")
            return None, label

    label = f"Uploaded: {uploaded.name}"
    try:
        return dh.read_uploaded_csv(uploaded.getvalue()), label
    except an.DataValidationError as exc:
        st.error(f"**Could not use `{uploaded.name}`.** {exc}")
        st.info(
            "Fix the file and upload it again, or remove it to go back to the sample data. "
            "Required columns: `order_date`, `product`, `region`, `revenue`, `cost`. "
            "Revenue and cost must be non-negative numbers."
        )
        return None, label


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

st.title("📊 BusinessLens AI")
st.caption(
    "A simple sales dashboard. Every figure is calculated by tested, deterministic "
    "Python code. Your CSV stays in this app's memory; only if you enable AI questions are "
    "your question and aggregated results sent to the provider you choose."
)

data, source = load_data()
if data is None:
    st.stop()

st.sidebar.success(source)

# ---- Filters (limits come from the data itself) ----
first_day, last_day = dh.date_bounds(data)
all_products = sorted(data["product"].unique())
all_regions = sorted(data["region"].unique())
data_key = f"{source}-{len(data)}-{first_day}-{last_day}"  # reset filters when the dataset changes

st.sidebar.header("Filters")
date_range = st.sidebar.date_input(
    "Date range",
    value=(first_day, last_day),
    min_value=first_day,
    max_value=last_day,
    key=f"dates-{data_key}",
)
products = st.sidebar.multiselect(
    "Products", all_products, default=all_products, key=f"products-{data_key}"
)
regions = st.sidebar.multiselect(
    "Regions", all_regions, default=all_regions, key=f"regions-{data_key}"
)

if not isinstance(date_range, tuple) or len(date_range) != 2:
    st.info("Pick an end date to complete the date range.")
    st.stop()

filtered = dh.filter_sales(data, date_range[0], date_range[1], products, regions)

# ---- Data preview (before filters, so users can check what was loaded) ----
with st.expander(f"Data preview: {len(data):,} rows loaded", expanded=False):
    st.dataframe(data.head(20), width="stretch", hide_index=True)
    st.caption(
        f"Showing the first 20 rows. Dates run from {first_day:%d %b %Y} to {last_day:%d %b %Y}."
    )

if filtered.empty:
    st.warning(
        "No sales match the current filters. Widen the date range or select at "
        "least one product and one region."
    )
    st.stop()

st.caption(
    f"Showing {len(filtered):,} of {len(data):,} rows "
    f"({date_range[0]:%d %b %Y} to {date_range[1]:%d %b %Y})."
)

# ---- Settings: AI provider (per browser session; the key is never written to disk) ----
st.subheader("Settings: AI provider")
PROVIDER_STATE = ("bl_provider", "bl_config_label", "bl_limiter", "bl_applied", "bl_status")


def reset_ai_settings():
    for key in (*PROVIDER_STATE, "bl_key_input"):
        st.session_state.pop(key, None)


def apply_settings(provider_id, model, api_key, base_url):
    """Validate the settings and keep them in this session only. Returns the provider or None."""
    try:
        provider = create_provider(ProviderConfig(provider_id, model, api_key, base_url or None))
    except ProviderError as exc:
        st.session_state["bl_status"] = ("error", str(exc))
        return None
    st.session_state["bl_applied"] = (provider_id, model.strip(), (base_url or "").strip())
    st.session_state["bl_provider"] = provider
    st.session_state["bl_config_label"] = f"{PROVIDERS[provider_id].label}, model `{model.strip()}`"
    st.session_state.setdefault("bl_limiter", ai.UsageLimiter())
    return provider


# Result of the last Save / Test (safe text only, never the key). It is drawn above the
# expander so it stays visible, at full width, even when the expander is collapsed.
status_slot = st.container()

with st.expander("Choose provider, model and API key", expanded="bl_provider" not in st.session_state):
    st.warning(
        "When you ask a question, your question text and the aggregated results of the "
        "analytics tools (totals, rankings, and product/region names) are sent to the cloud "
        "provider you select. Raw CSV rows are not sent. Providers differ in how they "
        "log, retain and train on data; read your provider's policy first."
    )
    provider_id = st.selectbox(
        "Provider", list(PROVIDERS), format_func=lambda p: PROVIDERS[p].label, key="bl_provider_id")
    info = PROVIDERS[provider_id]
    model = st.text_input("Model ID", value=info.default_model, key=f"bl_model_{provider_id}",
                          max_chars=128, help="Must support tool (function) calling.")
    api_key = st.text_input("API key", type="password", key="bl_key_input", max_chars=512,
                            autocomplete="off", help=info.key_help)
    base_url = ""
    if info.supports_base_url:
        base_url = st.text_input(
            "Base URL (optional)", key=f"bl_base_{provider_id}", max_chars=200,
            placeholder="https://api.example.com/v1",
            help="For OpenAI-compatible services only. https and public hosts only; "
                 "localhost, private networks and redirects are blocked. Leave empty for OpenAI.")
    st.caption("The key stays in this browser session's memory. It is not saved to disk and "
               "is cleared when you press Clear or the session ends.")
    # Equal columns that stack on narrow screens; nothing is sent until a button is pressed.
    save_col, test_col, clear_col = st.columns(3)
    save = save_col.button("Save", help="Use these settings for this session", width="stretch")
    test = test_col.button("Test connection", help="Sends one tiny request to the provider",
                           width="stretch")
    if clear_col.button("Clear key", width="stretch"):
        reset_ai_settings()
        st.rerun()
    if save or test:
        provider = apply_settings(provider_id, model, api_key, base_url)
        if provider is not None and test:
            try:
                with st.spinner("Testing..."):
                    ai.test_connection(provider, st.session_state["bl_limiter"])
                st.session_state["bl_status"] = (
                    "success", "Connection works and the model accepted the tool definitions.")
            except (ProviderError, ai.AgentError) as exc:
                st.session_state["bl_status"] = (
                    "error", f"Settings saved, but the connection test failed. {exc}")
        elif provider is not None:
            st.session_state["bl_status"] = ("success", "Saved for this session.")

with status_slot:
    applied = st.session_state.get("bl_applied")
    chosen = (provider_id, model.strip(), (base_url or "").strip())
    if applied is not None and chosen != applied:
        st.warning(
            f"**Pending, not saved yet:** {PROVIDERS[provider_id].label}, model `{chosen[1]}`. "
            f"Questions still use {st.session_state['bl_config_label']} until you press "
            "**Save** or **Test connection**."
        )
    status = st.session_state.get("bl_status")
    if status is not None:
        (st.success if status[0] == "success" else st.error)(status[1])

# ---- Ask a business question (the model calls the approved tools) ----
st.subheader("Ask a business question")
provider = st.session_state.get("bl_provider")
if provider is None:
    st.info("Add your provider, model and API key in Settings above to enable questions. "
            "The dashboard works without it.")
else:
    st.caption(
        f"Using {st.session_state['bl_config_label']}. Answers use the filtered data above. "
        "The model only calls approved, tested analytics tools; it does not do the math. "
        "Your question and the aggregated tool results are sent to your selected provider."
    )
    with st.form("ask"):
        question = st.text_input(
            "Question", placeholder="Which product has the best profit margin?",
            max_chars=ai.MAX_QUESTION_CHARS,
        )
        asked = st.form_submit_button("Ask")
    if asked:
        try:
            with st.spinner("Thinking..."):
                answer = ai.ask(question, filtered, provider, st.session_state["bl_limiter"])
            st.markdown(answer.text)
            if answer.tool_calls:
                with st.expander("Tools used"):
                    for call in answer.tool_calls:
                        st.code(f"{call['name']}({call['args']})" + ("" if call["ok"] else "  -> error"))
        except (ai.AgentError, ProviderError) as exc:
            st.error(str(exc))

# ---- KPI cards ----
margin = an.profit_margin(filtered)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total revenue", dh.format_number(an.total_revenue(filtered)))
kpi2.metric("Total cost", dh.format_number(an.total_cost(filtered)))
kpi3.metric("Total profit", dh.format_number(an.total_profit(filtered)))
kpi4.metric(
    "Profit margin",
    dh.format_percent(margin),
    help="Profit divided by revenue. Shown as N/A when revenue is zero.",
)
if margin is None:
    st.caption("Profit margin is not available because total revenue is zero.")

# ---- Revenue trend ----
st.subheader("Revenue trend")
period = st.radio(
    "Group by", ["month", "quarter", "year"], horizontal=True, label_visibility="collapsed"
)
trend = an.period_growth(filtered, period)
trend_fig = px.bar(
    trend, x="period", y="revenue", labels={"period": period.capitalize(), "revenue": "Revenue"}
)
trend_fig.update_traces(marker_color=BAR_COLOR)
trend_fig.update_xaxes(type="category")
trend_fig.update_layout(margin=dict(t=10, b=10))
st.plotly_chart(trend_fig, width="stretch")

with st.expander(f"Revenue and growth by {period}"):
    st.dataframe(
        trend.rename(columns={"previous_revenue": "previous period revenue"}),
        width="stretch",
        hide_index=True,
        column_config={
            "revenue": st.column_config.NumberColumn("Revenue", format="%.2f"),
            "previous period revenue": st.column_config.NumberColumn(
                "Previous period", format="%.2f"
            ),
            "growth": st.column_config.NumberColumn("Growth", format="percent"),
        },
    )
    st.caption(
        "Growth compares each period with the one before it. It is blank for the first "
        "period and when the previous period had no revenue. Periods with no sales show 0."
    )

# ---- Product and regional performance ----
products_df = an.product_performance(filtered)
regions_df = an.region_performance(filtered)

table_config = {
    "revenue": st.column_config.NumberColumn("Revenue", format="%.2f"),
    "cost": st.column_config.NumberColumn("Cost", format="%.2f"),
    "profit": st.column_config.NumberColumn("Profit", format="%.2f"),
    "profit_margin": st.column_config.NumberColumn("Profit margin", format="percent"),
}


def performance_chart(table, column, title):
    """Bar chart of revenue and profit (already calculated) for each product/region."""
    fig = px.bar(
        table,
        x=column,
        y=["revenue", "profit"],
        barmode="group",
        title=title,
        labels={column: column.capitalize(), "value": "Amount", "variable": ""},
        color_discrete_map={"revenue": BAR_COLOR, "profit": PROFIT_COLOR},
    )
    fig.update_layout(margin=dict(t=50, b=10), legend=dict(orientation="h", y=-0.2))
    return fig


left, right = st.columns(2)
with left:
    st.plotly_chart(performance_chart(products_df, "product", "Revenue and profit by product"), width="stretch")
with right:
    st.plotly_chart(performance_chart(regions_df, "region", "Revenue and profit by region"), width="stretch")

st.subheader("Performance tables")
tab_products, tab_regions = st.tabs(["By product", "By region"])
with tab_products:
    st.dataframe(products_df, width="stretch", hide_index=True, column_config=table_config)
with tab_regions:
    st.dataframe(regions_df, width="stretch", hide_index=True, column_config=table_config)
st.caption("Sorted by revenue, highest first. Profit margin is blank where revenue is zero.")
