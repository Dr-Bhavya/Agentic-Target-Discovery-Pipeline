import os
import streamlit as st
import requests
import networkx as nx
import pandas as pd
import io
from phi.agent import Agent
from phi.model.groq import Groq

# -----------------------------------------------------------------------------
# 1. STREAMLIT GLOBAL LAYOUT & CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DAVID Multi-Agent Target Funnel", layout="wide", page_icon="🧬")

st.title("🧬 TargetScout-AI: Systems Biology Funnel (DAVID + Groq)")
st.caption("Integrated Multi-Agent Pipeline via Live STRING DB, DAVID API, and Phidata Orchestration")

# Initialize session state cache blocks safely
if "topology_df" not in st.session_state:
    st.session_state["topology_df"] = None
if "network_obj" not in st.session_state:
    st.session_state["network_obj"] = None

# Sidebar Controls
with st.sidebar:
    st.header("🔑 Configuration")
    user_api_key = st.text_input("Enter Free Groq API Key:", type="password")
    st.markdown("[Get a free Groq API key here](https://groq.com)")
    
    st.header("📊 Network Parameters")
    confidence_score = st.slider("STRING Confidence Cutoff", 150, 900, 400, step=50)
    
    st.header("🎛️ Topology Filter Switch")
    topological_cutoff = st.slider("Consensus Score Threshold (Cutoff)", 0.0, 1.0, 0.35, step=0.05)
    st.caption("Genes scoring below this mathematical average are classified as peripheral nodes.")

# Securely bind the Groq environment variable if provided
if user_api_key:
    os.environ["GROQ_API_KEY"] = user_api_key

# -----------------------------------------------------------------------------
# 2. BIOINFORMATICS PIPELINES (STRING NETWORK & DAVID API)
# -----------------------------------------------------------------------------
def run_network_topology_pipeline(genes: list) -> dict:
    """Hits the live STRING API, builds networks via NetworkX, and runs centralities."""
    url = "https://string-db.org"
    payload = {
        "identifiers": "\n".join(genes), 
        "species": 9606, 
        "required_score": confidence_score, 
        "caller_identity": "targetscout_ai"
    }
    
    G = nx.Graph()
    try:
        response = requests.post(url, data=payload, timeout=12)
        if response.status_code == 200:
            interactions = response.json()
            for edge in interactions:
                p1 = edge.get("preferredName_A") or edge.get("stringId_A")
                p2 = edge.get("preferredName_B") or edge.get("stringId_B")
                score = edge.get("score")
                if p1 and p2:
                    G.add_edge(p1, p2, weight=score)
    except Exception as e:
        st.error(f"Error fetching network parameters: {e}")
        
    for g in genes:
        if g not in G: 
            G.add_node(g)
            
    # Calculate Node Centralities
    deg_cent = nx.degree_centrality(G)
    bet_cent = nx.betweenness_centrality(G) if len(G.nodes()) > 2 else {n: 0.0 for n in G.nodes()}
    clo_cent = nx.closeness_centrality(G)
    
    metrics = [{
        "Gene": node,
        "Degree Centrality": round(deg_cent[node], 4),
        "Betweenness Centrality": round(bet_cent[node], 4),
        "Closeness Centrality": round(clo_cent[node], 4),
    } for node in G.nodes()]
    
    df = pd.DataFrame(metrics)
    df["Hybrid Combined Score"] = (df["Degree Centrality"] + df["Betweenness Centrality"] + df["Closeness Centrality"]) / 3
    df["Hybrid Combined Score"] = df["Hybrid Combined Score"].round(4)
    df = df.sort_values(by="Hybrid Combined Score", ascending=False).reset_index(drop=True)
    
    df["Status"] = df["Hybrid Combined Score"].apply(
        lambda x: "Influential Gene Hub" if x >= topological_cutoff else "Peripheral Node"
    )
    
    st.session_state["topology_df"] = df
    st.session_state["network_obj"] = G
    return {"status": "success", "df": df, "graph": G}


def run_david_enrichment_pipeline(genes: list) -> tuple:
    """Submits genes programmatically to DAVID API and parses KEGG and OMIM charts."""
    # Correct Live Programmatic Request URL for DAVID 
    david_url = "https://nih.gov"
    
    payload = {
        "type": "OFFICIAL_GENE_SYMBOL",
        "ids": ",".join(genes),
        "tool": "chartReport",
        "annot": "KEGG_PATHWAY,OMIM_DISEASE"
    }
    
    kegg_rows = []
    omim_rows = []
    
    try:
        response = requests.get(david_url, params=payload, timeout=15)
        if response.status_code == 200 and "html" not in response.text.lower():
            # DAVID API returns tab-separated chart data text matrices
            lines = response.text.strip().split("\n")
            if len(lines) > 1:
                for line in lines[1:]: # Skip header
                    row = line.split("\t")
                    if len(row) < 5:
                        continue
                    
                    category = row[0].strip()
                    term = row[1].strip()
                    count = row[2].strip()
                    p_val = float(row[4])
                    mapped_genes = row[5].strip() if len(row) > 5 else ""
                    
                    data_dict = {
                        "Term": term,
                        "Gene Count": count,
                        "P-Value": p_val,
                        "Mapped Genes": [g.strip().upper() for g in mapped_genes.split(",") if g.strip()]
                    }
                    
                    if "KEGG" in category:
                        kegg_rows.append(data_dict)
                    elif "OMIM" in category or "DISEASE" in category:
                        omim_rows.append(data_dict)
    except Exception as e:
        st.error(f"Error calling DAVID Web Services API: {e}")
        
    return pd.DataFrame(kegg_rows), pd.DataFrame(omim_rows)

# -----------------------------------------------------------------------------
# 3. INTERACTIVE STREAMLIT APPLICATION INTERFACE FLOW
# -----------------------------------------------------------------------------
gene_input = st.text_area(
    "🧬 Enter Gene Symbols List (One Gene Symbol per line)", 
    value="TP53\nEGFR\nSTAT3\nAKT1\nBRCA1\nMYC\nTNF\nIL6\nVEGFA\nPTEN", 
    height=150
)
genes = [g.strip().upper() for g in gene_input.split("\n") if g.strip()]

if st.button("🚀 Run Multi-Agent Analytical Pipeline"):
    if not user_api_key:
        st.warning("⚠️ Access Denied: Please provide a valid Groq API Key on your sidebar parameters to trigger the AI.")
        st.stop()
    if not genes:
        st.warning("⚠️ Input Required: Please supply structural gene references.")
        st.stop()
        
    with st.spinner("Processing workflows via live STRING & DAVID framework engines..."):
        topo_result = run_network_topology_pipeline(genes)
        kegg_df, omim_df = run_david_enrichment_pipeline(genes)
        
    if topo_result["status"] == "success":
        topo_df = topo_result["df"]
        G = topo_result["graph"]
        
        influential_genes = topo_df[topo_df["Status"] == "Influential Gene Hub"]["Gene"].tolist()
        
        tab1, tab2, tab3 = st.tabs(["📊 Network Topology", "🔬 DAVID Enrichment Analytics", "📚 Multi-Agent Literature Survey"])
        
        # --- TAB 1: GRAPH TOPOLOGY VIEW ---
        with tab1:
            st.header("🕸️ Protein Interaction Network & Topological Ranks")
            
            col1, col2 = st.columns()
            with col1:
                st.subheader("📌 Calculated Parameters & Hybrid Rankings")
                st.dataframe(topo_df, use_container_width=True)
            with col2:
                st.subheader("🎯 Identified Influential Genes")
                if influential_genes:
                    for ig in influential_genes:
                        st.info(f"✨ **{ig}** satisfies criteria as an influential hub network target.")
                else:
                    st.warning("No genes scaled past your specific Consensus Score Threshold selector.")
                    
            st.subheader("🔗 Network Interaction Edges")
            edges_data = [{"Source": u, "Target": v, "STRING Score": d.get("weight", 0)} for u, v, d in G.edges(data=True)]
            if edges_data:
                st.dataframe(pd.DataFrame(edges_data).sort_values(by="STRING Score", ascending=False), use_container_width=True)
            else:
                st.info("No connections map above your threshold choice.")

        # --- TAB 2: FUNCTIONAL ENRICHMENT DATA VIA DAVID ---
        with tab2:
            st.header("🧪 DAVID Functional Annotation Charts")
            
            def map_influential_genes(df):
                if df.empty: return pd.DataFrame()
                df['Influential Genes Mapped'] = df['Mapped Genes'].apply(
                    lambda x: ", ".join([ig for ig in influential_genes if ig in x]) if isinstance(x, list) else ""
                )
                # Convert list back to displayable string format
                df['Mapped Genes'] = df['Mapped Genes'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
                return df.sort_values(by="P-Value")

            col_kegg, col_omim = st.columns(2)
            with col_kegg:
                st.subheader("🗺️ Significant KEGG Pathway Mapping")
                formatted_kegg = map_influential_genes(kegg_df)
                if not formatted_kegg.empty:
                    st.dataframe(formatted_kegg, use_container_width=True)
                else:
                    st.info("No explicit KEGG data matched or returned by DAVID.")
                    
            with col_omim:
                st.subheader("🏥 OMIM Disease Profiling Matrix")
                formatted_omim = map_influential_genes(omim_df)
                if not formatted_omim.empty:
                    st.dataframe(formatted_omim, use_container_width=True)
                else:
            st.info("No explicit OMIM profiles matched or returned by DAVID.")

        # --- TAB 3: AGENTIC INTERPRETATION MATRIX ---
        with tab3:
            st.header("🤖 Phidata Systems Biology AI Agent Survey")
            
            topo_context_str = topo_df.to_string()
            kegg_context_str = formatted_kegg.head(10).to_string() if not formatted_kegg.empty else "No significant pathways."
            omim_context_str = formatted_omim.head(10).to_string() if not formatted_omim.empty else "No significant diseases."
            
            if not influential_genes:
                st.error("Cannot execute AI survey because no genes were isolated as an 'Influential Gene Hub'. Lower your consensus cutoff.")
            else:
                with st.spinner("Orchestrating agent collaboration spaces..."):
                    
                    # Agent 1: The Graph Data Analyst
                    analyst_agent = Agent(
                        name="Bioinformatics Systems Analyst",
                        model=Groq(id="llama-3.3-70b-versatile"),
                        description="You are a senior bioinformatician specialized in analyzing network topology maps from protein interaction matrices.",
                        instructions=[
                            "Review the provided topological structural statistics text block.",
                            f"Focus entirely on analyzing these identified influential genes: {', '.join(influential_genes)}.",
                            "Explain how their high Degree, Betweenness, or Closeness Centrality scores render them crucial communication nodes."
                        ],
                        markdown=True
                    )
                    
                    # Agent 2: The Medical Literature Curation Agent
                    literature_agent = Agent(
                        name="Medical Literature Architect",
                        model=Groq(id="llama-3.3-70b-versatile"),
                        description="You are a molecular geneticist specialized in translating mathematical network values into real mechanistic clinical context.",
                        instructions=[
                            "Review the evaluation provided by the Systems Analyst.",
                            "Cross-reference the isolated hub targets against the provided DAVID KEGG pathways and OMIM disease matrices.",
                            "Provide a comprehensive, publication-ready literature review detailing the exact molecular mechanisms connecting these specific hub genes to those diseases and pathways."
                        ],
                        markdown=True
                    )
                    
                    # Orchestrate Task Execution Flow
                    st.subheader("📝 Analyst Report: Network Significance")
                    analyst_response = analyst_agent.run(f"Analyze this network topology data matrix:\n{topo_context_str}")
                    st.markdown(analyst_response.content)
                    
                    st.markdown("---")
                    
                    st.subheader("📚 Curated Literature Synthesis")
                    literature_prompt = (
                        f"Based on the analyst's profile, look at these biological metrics profiles:\n\n"
                        f"Top DAVID KEGG Context:\n{kegg_context_str}\n\n"
                        f"Top DAVID OMIM Disease Context:\n{omim_context_str}\n\n"
                        f"Synthesize structural mechanics for the target genes: {', '.join(influential_genes)}."
                    )
                    literature_response = literature_agent.run(literature_prompt)
                    st.markdown(literature_response.content)
