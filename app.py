# Multi-Agent Systems Biology Target Funnel
#
# pip install streamlit requests networkx pandas matplotlib agno groq
#
# Run with: streamlit run app.py

import os
import re
import time

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
import streamlit as st
from agno.agent import Agent
from agno.models.groq import Groq

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Agent Target Discovery Studio", layout="wide")
st.title("🧬 Multi-Agent Systems Biology Target Funnel")
st.caption("Powered by Groq API, STRING DB, Enrichr & NCBI PubMed — an automated target prioritization workflow.")

for key in ["topology_df", "network_obj", "pathway_df", "disease_df", "influential_targets"]:
    if key not in st.session_state:
        st.session_state[key] = None

with st.sidebar:
    st.header("🔑 Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    groq_model = st.selectbox("Select Groq Model:", ["qwen/qwen3.6-27b", "openai/gpt-oss-120b"])
    st.markdown("[Get your Groq API key here](https://console.groq.com/keys)")

    st.header("📊 Network Parameters")
    confidence_score = st.slider("STRING Confidence Cutoff", 150, 900, 400, step=50)
    st.caption(
        "Influential (hub) genes are identified automatically from Degree, Betweenness, and Closeness "
        "Centrality — thresholds are derived from each metric's own mean + 1 standard deviation, no manual "
        "cutoff needed."
    )

STRING_API_URL = "https://string-db.org/api/json/network"
ENRICHR_ADD_URL = "https://maayanlab.cloud/Enrichr/addList"
ENRICHR_ENRICH_URL = "https://maayanlab.cloud/Enrichr/enrich"
EUTILS_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def parse_gene_list(gene_list_str: str) -> list:
    return sorted(set(g.strip().upper() for g in gene_list_str.replace(",", "\n").split("\n") if g.strip()))


def compute_topology_metrics(G: nx.Graph) -> pd.DataFrame:
    """Computes degree, betweenness, and closeness centrality for every node."""
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G) if G.number_of_nodes() > 2 else {n: 0.0 for n in G.nodes()}
    clo_cent = nx.closeness_centrality(G)

    df = pd.DataFrame([{
        "Gene": node,
        "Degree Centrality": round(deg_cent[node], 4),
        "Betweenness Centrality": round(bet_cent[node], 4),
        "Closeness Centrality": round(clo_cent[node], 4),
    } for node in G.nodes()])
    return df


def flag_influential_genes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically identifies influential (hub) genes with no user-supplied cutoff.

    For each of the three centrality metrics independently, the threshold is derived
    from the network's own distribution: mean + 1 standard deviation — the standard
    statistical definition of a "hub" node in network biology. A gene is flagged as
    influential if it clears that self-derived threshold on at least 2 of the 3 metrics
    (majority vote), which is more robust than requiring unanimity or relying on a
    single blended score.
    """
    metrics = ["Degree Centrality", "Betweenness Centrality", "Closeness Centrality"]
    votes = pd.Series(0, index=df.index)
    thresholds = {}

    for metric in metrics:
        mean, std = df[metric].mean(), df[metric].std(ddof=0)
        threshold = mean + std
        thresholds[metric] = round(threshold, 4)
        votes += (df[metric] > threshold).astype(int)

    df = df.copy()
    df["Hub Votes (of 3)"] = votes
    df["Influential"] = votes >= 2

    fallback_note = None
    if not df["Influential"].any():
        # Distribution too flat / network too small for mean+SD to separate anyone (e.g. a
        # near-complete or very small graph). Relax to "clears the bar on at least one metric".
        df["Influential"] = votes >= 1
        fallback_note = "no gene cleared 2 of 3 auto-thresholds; relaxed to 1 of 3"

    if not df["Influential"].any():
        # Still nothing (e.g. a fully symmetric graph where every node is tied). Fall back to
        # the single top-ranked gene by summed centrality so the pipeline can still proceed.
        top_idx = (df[metrics].sum(axis=1)).idxmax()
        df.loc[top_idx, "Influential"] = True
        fallback_note = "network too symmetric for statistical thresholds; selected the single top-ranked gene"

    df = df.sort_values(by=metrics, ascending=False).reset_index(drop=True)
    df.attrs["thresholds"] = thresholds
    df.attrs["fallback_note"] = fallback_note
    return df


# --------------------------------------------------------------------------------------
# Agent 1: Network topology (STRING)
# --------------------------------------------------------------------------------------
def run_network_topology_pipeline(gene_list_str: str, required_score: int) -> dict:
    """Builds a STRING-only interaction network and computes centrality metrics."""
    genes = parse_gene_list(gene_list_str)
    if not genes:
        return {"status": "error", "message": "No valid gene symbols provided."}

    G = nx.Graph()
    G.add_nodes_from(genes)

    payload = {
        "identifiers": "\n".join(genes),
        "species": 9606,
        "required_score": required_score,
        "add_nodes": 0,  # explicitly forbid STRING from adding extra interactor nodes
        "caller_identity": "multi_agent_bio_workflow",
    }

    try:
        response = requests.post(STRING_API_URL, data=payload, timeout=20)
        response.raise_for_status()
        interactions = response.json()
    except (requests.RequestException, ValueError) as exc:
        return {
            "status": "error",
            "message": f"STRING API request failed ({exc}). Please retry — no simulated data is shown "
                        f"for a research tool like this.",
        }

    if isinstance(interactions, dict) and interactions.get("Error"):
        return {"status": "error", "message": f"STRING API error: {interactions['Error']}"}

    gene_set = set(genes)
    skipped_extra_nodes = set()
    for edge in interactions:
        p1 = edge.get("preferredName_A", "").upper()
        p2 = edge.get("preferredName_B", "").upper()
        score = edge.get("score")
        if not p1 or not p2:
            continue
        # Belt-and-suspenders: only keep edges strictly between genes the user actually supplied.
        # STRING's identifier resolution can occasionally surface a synonym/alias node instead of
        # the exact symbol queried, so this guards against silently growing the gene set.
        if p1 not in gene_set or p2 not in gene_set:
            skipped_extra_nodes.update({p1, p2} - gene_set)
            continue
        G.add_edge(p1, p2, weight=score)

    df = compute_topology_metrics(G)
    df = flag_influential_genes(df)

    edge_count = G.number_of_edges()
    thresholds = df.attrs.get("thresholds", {})
    threshold_str = ", ".join(f"{k} > {v}" for k, v in thresholds.items())
    message = (f"Live STRING network parsed successfully — {edge_count} interaction(s) found "
               f"at confidence ≥ {required_score}, restricted strictly to your {len(genes)} input gene(s). "
               f"Auto-derived hub thresholds (mean + 1 SD): {threshold_str}.")
    if skipped_extra_nodes:
        message += (f" Note: STRING returned {len(skipped_extra_nodes)} additional node(s) not in your "
                     f"input list ({', '.join(sorted(skipped_extra_nodes))}) — these were excluded.")
    if df.attrs.get("fallback_note"):
        message += f" ⚠️ Fallback applied: {df.attrs['fallback_note']}."

    return {
        "status": "success",
        "df": df,
        "graph": G,
        "message": message,
    }


# --------------------------------------------------------------------------------------
# Agent 2: Functional enrichment (Enrichr — KEGG pathways + OMIM diseases)
# --------------------------------------------------------------------------------------
def run_enrichment_pipeline(gene_list_str: str, influential_genes: list) -> dict:
    """Submits all genes to Enrichr and extracts significant KEGG pathway and OMIM disease terms."""
    genes = parse_gene_list(gene_list_str)
    influential_set = set(influential_genes)

    try:
        add_resp = requests.post(
            ENRICHR_ADD_URL,
            files={
                "list": (None, "\n".join(genes)),
                "description": (None, "multi_agent_bio_workflow"),
            },
            timeout=20,
        )
        add_resp.raise_for_status()
        user_list_id = add_resp.json()["userListId"]
    except (requests.RequestException, ValueError, KeyError) as exc:
        empty = pd.DataFrame(columns=["Description", "P-Value", "Influential Genes Mapped"])
        return {
            "kegg_df": empty, "disease_df": empty,
            "top_pathway": None, "top_disease": None,
            "message": f"Enrichr submission failed ({exc}). No enrichment data available — please retry.",
        }

    def fetch_library(library: str) -> pd.DataFrame:
        try:
            resp = requests.get(
                ENRICHR_ENRICH_URL,
                params={"userListId": user_list_id, "backgroundType": library},
                timeout=20,
            )
            resp.raise_for_status()
            rows = resp.json().get(library, [])
        except (requests.RequestException, ValueError):
            return pd.DataFrame(columns=["Description", "P-Value", "Influential Genes Mapped"])

        records = []
        for row in rows[:15]:
            # Enrichr row layout: [rank, term, p-value, z-score, combined-score,
            #                      overlapping genes, adjusted p-value, old p-value, old adjusted p-value]
            term_name = row[1]
            p_value = row[2]
            overlap_genes = [g.strip().upper() for g in row[5]]
            mapped = [g for g in overlap_genes if g in influential_set]
            records.append({
                "Description": term_name,
                "P-Value": p_value,
                "Influential Genes Mapped": ", ".join(mapped) if mapped else "None",
            })

        if not records:
            return pd.DataFrame(columns=["Description", "P-Value", "Influential Genes Mapped"])
        return pd.DataFrame(records).sort_values(by="P-Value").reset_index(drop=True)

    kegg_df = fetch_library("KEGG_2021_Human")
    disease_df = fetch_library("OMIM_Disease")

    top_pathway = kegg_df.iloc[0]["Description"] if not kegg_df.empty else None
    top_disease = disease_df.iloc[0]["Description"] if not disease_df.empty else None

    return {
        "kegg_df": kegg_df,
        "disease_df": disease_df,
        "top_pathway": top_pathway,
        "top_disease": top_disease,
        "message": "Live enrichment results parsed from Enrichr (KEGG_2021_Human, OMIM_Disease).",
    }


# --------------------------------------------------------------------------------------
# Agent 3: Literature evidence (NCBI PubMed E-utilities)
# --------------------------------------------------------------------------------------
def run_pubmed_literature_pipeline(target_gene: str, pathway: str, disease: str) -> str:
    """Queries NCBI E-utilities for abstracts connecting a gene to the top pathway/disease."""
    gene = target_gene.upper().strip()
    context_term = pathway or disease or "therapeutic target"
    clean_context = re.sub(r"[^A-Za-z0-9 ]", " ", context_term).strip()

    def esearch(term: str) -> list:
        params = {
            "db": "pubmed", "term": term, "retmode": "json", "retmax": "2",
            "tool": "MultiAgentBioWorkflow", "email": "research@example.com",
        }
        r = requests.get(EUTILS_ESEARCH_URL, params=params, timeout=15)
        r.raise_for_status()
        return r.json().get("esearchresult", {}).get("idlist", [])

    try:
        term = f"{gene}[Title/Abstract] AND ({clean_context}[Title/Abstract] OR therapeutic target)"
        id_list = esearch(term)

        if not id_list:
            id_list = esearch(f"{gene}[Title/Abstract] AND therapeutic target")

        if not id_list:
            return f"- **{gene}**: No PubMed abstracts found linking this gene to the enriched pathway/disease context."

        summary_resp = requests.get(
            EUTILS_ESUMMARY_URL,
            params={"db": "pubmed", "id": ",".join(id_list), "retmode": "json"},
            timeout=15,
        )
        summary_resp.raise_for_status()
        summary_results = summary_resp.json().get("result", {})

        citations = []
        for i, pmid in enumerate(id_list, start=1):
            info = summary_results.get(pmid, {})
            title = info.get("title", "Untitled record")
            pub_date = info.get("pubdate", "n.d.")
            journal = info.get("source", "PubMed")
            citations.append(f"  [{i}] {title} — **{journal}** ({pub_date}). [PubMed link](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

        return f"- **{gene}** literature evidence:\n" + "\n".join(citations)

    except (requests.RequestException, ValueError) as exc:
        return f"- **{gene}**: PubMed lookup failed ({exc}). No literature evidence retrieved — please retry."


# --------------------------------------------------------------------------------------
# UI: input + orchestration
# --------------------------------------------------------------------------------------
default_genes = "SERPINE1\nMMP1\nMMP7\nTGFB1\nEGFR\nSTAT3\nVEGFA\nIL6\nAKT1"
input_genes = st.text_area("Provide Gene Symbols (One Gene per Line):", value=default_genes, height=180)

if st.button("🚀 Launch Autonomous Target Prioritization Pipeline"):
    if not groq_api_key:
        st.error("Please add your Groq API Key in the sidebar to authenticate the agents.")
        st.stop()

    os.environ["GROQ_API_KEY"] = groq_api_key

    with st.status("🕵️ Orchestrating Multi-Agent Discovery Pipeline across Omics Layers...", expanded=True) as status:
        st.write("1. 🕸️ Network Analyst Agent → mapping STRING nodes & centralities...")
        net_results = run_network_topology_pipeline(input_genes, confidence_score)
        if net_results["status"] == "error":
            status.update(label="❌ Pipeline halted", state="error")
            st.error(net_results["message"])
            st.stop()
        st.write(f"   {net_results['message']}")

        topology_df = net_results["df"]
        network_graph = net_results["graph"]
        st.session_state["topology_df"] = topology_df
        st.session_state["network_obj"] = network_graph

        st.write("2. 🎛️ Selecting influential (hub) genes via auto-derived centrality thresholds...")
        surviving_df = topology_df[topology_df["Influential"]].reset_index(drop=True)
        influential_targets = surviving_df["Gene"].tolist()
        st.session_state["influential_targets"] = influential_targets
        st.write(f"   Influential genes ({len(influential_targets)}): {', '.join(influential_targets)}")

        st.write("3. 🧬 Enrichment Analyst Agent → querying Enrichr for KEGG & OMIM terms...")
        enrich_results = run_enrichment_pipeline(input_genes, influential_targets)
        st.session_state["pathway_df"] = enrich_results["kegg_df"]
        st.session_state["disease_df"] = enrich_results["disease_df"]
        top_pathway_found = enrich_results["top_pathway"] or "no significantly enriched pathway"
        top_disease_found = enrich_results["top_disease"] or "no significantly enriched disease"
        st.write(f"   {enrich_results['message']}")

        st.write("4. 📚 Literature Miner Agent → querying NCBI PubMed...")
        literature_payload_items = []
        for target in influential_targets[:4]:
            st.write(f"   • Mining literature for: {target}")
            literature_payload_items.append(
                run_pubmed_literature_pipeline(target, enrich_results["top_pathway"], enrich_results["top_disease"])
            )
            time.sleep(0.4)
        combined_lit_context = "\n\n".join(literature_payload_items)

        st.write("5. 🧠 Lead Orchestrator Agent → synthesizing executive dossier...")
        orchestrator_agent = Agent(
            name="Biomedical Discovery Lead",
            model=Groq(id=groq_model),
            instructions=[
                "You are a Senior Lead Translational Biologist coordinating structural bioinformatics data.",
                "Analyze the provided parameters, mapping data, and literature citations into an executive target blueprint.",
                "Break the output into three sections: 1. Graph Structural Insights, 2. Overlap Alignment Analysis, "
                "3. Literature Evidence Synthesis.",
                "Only state findings that are explicitly supported by the provided data — do not invent pathways, "
                "diseases, or citations that are not present in the input payload.",
                "If a pathway, disease, or literature item is marked as unavailable, say so plainly instead of "
                "filling in a plausible-sounding substitute.",
                "Explain how the influential genes cross-map into the enriched pathways and diseases provided.",
                "Do NOT re-type, reformat, or paraphrase the individual PubMed citation lines or their links from "
                "the literature evidence payload — a verbatim, correctly-linked reference list is appended "
                "separately after your response. In your 'Literature Evidence Synthesis' section, only summarize "
                "what the findings show and refer to sources by gene name (e.g. 'as shown in the EGFR literature "
                "below'), without repeating titles, journals, or URLs yourself.",
                "Maintain precise language fit for a biomedical research report dashboard.",
            ],
            markdown=True,
        )

        agent_prompt = f"""
        Synthesize these systems biology results into an Executive Target Dossier:
        [INPUT TARGET LIST]: {', '.join(parse_gene_list(input_genes))}
        [INFLUENTIAL TARGET NODES]: {', '.join(influential_targets)}
        [TOP ENRICHED PATHWAY]: {top_pathway_found}
        [TOP ENRICHED OMIM DISEASE]: {top_disease_found}

        [LITERATURE EVIDENCE PAYLOAD]:
        {combined_lit_context}
        """

        try:
            agent_response = orchestrator_agent.run(agent_prompt)
            master_dossier_text = (
                agent_response.content
                + "\n\n---\n\n### 📚 Literature References\n\n"
                + combined_lit_context
            )
            status.update(label="✅ Pipeline executed successfully!", state="complete")
        except Exception as e:
            status.update(label="⚠️ Summary generation interrupted", state="error")
            master_dossier_text = (
                f"An LLM API error occurred: {e}\n\nReview the structural charts in the tabs below.\n\n"
                f"---\n\n### 📚 Literature References\n\n{combined_lit_context}"
            )

    # ----------------------------------------------------------------------------------
    # Results layout
    # ----------------------------------------------------------------------------------
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "📋 Executive Target Dossier & Literature",
        "📊 Network Topology Studio",
        "🧬 Multi-Omics Enrichment Studio",
    ])

    with tab1:
        st.subheader("📋 Consolidated Master Target Dossier & Literature Summary")
        st.markdown(master_dossier_text)

    with tab2:
        st.subheader("📊 Network & Topology Analysis")
        col1, col2 = st.columns([4, 5])

        with col1:
            st.markdown("**🕸️ STRING Network View**")
            if network_graph is not None and network_graph.number_of_nodes() > 0:
                fig, ax = plt.subplots(figsize=(6, 5), facecolor="none")
                pos = nx.spring_layout(network_graph, k=0.4, seed=42)

                node_colors = ["#00E5FF" if n in influential_targets else "#E0E0E0" for n in network_graph.nodes()]
                node_sizes = [800 if n in influential_targets else 350 for n in network_graph.nodes()]

                nx.draw_networkx_nodes(network_graph, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                                        edgecolors="#112233", linewidths=1.5)
                nx.draw_networkx_edges(network_graph, pos, ax=ax, edge_color="#CFD8DC", width=1.5)
                nx.draw_networkx_labels(network_graph, pos, ax=ax, font_size=9, font_weight="bold",
                                         font_family="sans-serif")

                ax.axis("off")
                st.pyplot(fig)
                st.caption("🔵 Cyan = influential (passed cutoff). ⚪ Grey = filtered out by cutoff.")
            else:
                st.info("No interactions were returned by STRING for this gene set at the chosen confidence level.")

        with col2:
            st.markdown("**📈 Centrality Matrix**")
            if st.session_state["topology_df"] is not None:
                def highlight_survivors(row):
                    color = "background-color: rgba(0, 229, 255, 0.12)" if row["Influential"] else ""
                    return [color] * len(row)

                styled_df = st.session_state["topology_df"].style.apply(highlight_survivors, axis=1)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                st.caption(
                    "Rows highlighted in cyan are influential (hub) genes — auto-flagged by clearing the "
                    "mean + 1 SD threshold on at least 2 of the 3 centrality metrics."
                )

    with tab3:
        st.subheader("🧬 Enrichment & Influential Target Mapping")
        st.caption("Live functional enrichment results from Enrichr.")

        enc_col1, enc_col2 = st.columns(2)

        with enc_col1:
            st.markdown("**🌿 KEGG Pathway Enrichment**")
            if st.session_state.get("pathway_df") is not None and not st.session_state["pathway_df"].empty:
                st.dataframe(st.session_state["pathway_df"], use_container_width=True, hide_index=True)
            else:
                st.info("No significant KEGG pathway terms were returned for this gene set.")

        with enc_col2:
            st.markdown("**🏥 OMIM Disease Enrichment**")
            if st.session_state.get("disease_df") is not None and not st.session_state["disease_df"].empty:
                st.dataframe(st.session_state["disease_df"], use_container_width=True, hide_index=True)
            else:
                st.info("No significant OMIM disease terms were returned for this gene set.")
