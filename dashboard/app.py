"""Integrity review dashboard.

Run:
    streamlit run dashboard/app.py -- --data data/run2
"""
import argparse
import json
import os

import altair as alt
import polars as pl
import streamlit as st

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/run2")
args, _ = ap.parse_known_args()
D = args.data

st.set_page_config(page_title="Abuse Detection", layout="wide")
st.title("Behavioral Abuse Detection — review console")


@st.cache_data
def load():
    users = pl.read_parquet(os.path.join(D, "users.parquet"))
    dec = pl.read_parquet(os.path.join(D, "fusion", "decisions.parquet"))
    seq = pl.read_parquet(os.path.join(D, "seq", "scores.parquet"))
    mets = {}
    for name, p in [("xgb", "baseline"), ("seq", "seq"), ("gnn", "gnn"),
                    ("rings", "rings"), ("spray", "spray"), ("fusion", "fusion")]:
        f = os.path.join(D, p, "metrics.json")
        if os.path.exists(f):
            mets[name] = json.load(open(f))
    return users, dec, seq, mets


users, dec, seq, mets = load()
tab_ov, tab_queue, tab_ato = st.tabs(["overview", "enforcement queue", "ATO timeline"])

with tab_ov:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("population")
        st.dataframe(users.group_by("label").len().sort("label").to_pandas(),
                     hide_index=True)
    with c2:
        st.subheader("abuse PR-AUC by detector (held-out eval)")
        if "fusion" in mets:
            st.dataframe(
                [{"detector": k, "PR-AUC": v}
                 for k, v in mets["fusion"]["abuse_pr_auc"].items()],
                hide_index=True)
        if "fusion" in mets:
            st.caption(f'expected cost {mets["fusion"]["expected_cost"]:,} vs '
                       f'do-nothing {mets["fusion"]["cost_do_nothing"]:,}')
    st.subheader("fused risk by true label (eval half)")
    pdf = dec.filter(pl.col("is_eval")).to_pandas()
    st.altair_chart(alt.Chart(pdf).mark_boxplot().encode(
        x=alt.X("label:N"), y=alt.Y("risk:Q"),
        color="label:N").properties(height=300), use_container_width=True)

with tab_queue:
    st.subheader("tiered enforcement queue")
    tier = st.selectbox("tier", ["suspend", "challenge", "rate_limit"])
    q = (dec.filter(pl.col("tier") == tier)
         .join(users.select("user_id", "username", "subtype", "ring_id"),
               on="user_id")
         .sort("risk", descending=True)
         .select("user_id", "username", "risk", "label", "subtype",
                 "ring_id", "evasion"))
    st.caption(f"{len(q):,} accounts (label column = ground truth, "
               "shown for demo only)")
    st.dataframe(q.head(300).to_pandas(), hide_index=True)

with tab_ato:
    st.subheader("token-theft anomaly timeline")
    thieves = (users.filter(pl.col("label") == "token_thief")["user_id"]
               .to_list())
    uid = st.selectbox("account", thieves)
    ev = (pl.scan_parquet(os.path.join(D, "events.parquet"))
          .filter(pl.col("user_id") == uid)
          .select("ts", "tokens_used", "device_fp", "country")
          .collect())
    hourly = (ev.with_columns(hour=(pl.col("ts") // 3600 * 3600))
              .group_by("hour").agg(events=pl.len()).sort("hour")
              .with_columns(t=pl.from_epoch("hour")))
    chart = alt.Chart(hourly.to_pandas()).mark_line().encode(
        x="t:T", y="events:Q").properties(height=250)
    meta = users.filter(pl.col("user_id") == uid)
    srow = seq.filter(pl.col("user_id") == uid)
    overlays = [chart]
    take = meta["takeover_ts"][0]
    if take is not None:
        overlays.append(alt.Chart(alt.Data(values=[{"t": take * 1000}]))
                        .mark_rule(color="red", strokeWidth=2).encode(x="t:T"))
    if len(srow) and srow["seq_anomaly_ts"][0] > 0:
        overlays.append(alt.Chart(
            alt.Data(values=[{"t": int(srow["seq_anomaly_ts"][0]) * 1000}]))
            .mark_rule(color="orange", strokeDash=[6, 3], strokeWidth=2)
            .encode(x="t:T"))
    st.altair_chart(alt.layer(*overlays), use_container_width=True)
    st.caption("red = true takeover; orange dashed = detected anomaly spike; "
               f"evasion={meta['evasion'][0]:.2f}, "
               f"devices={ev['device_fp'].n_unique()}, "
               f"countries={ev['country'].n_unique()}")
