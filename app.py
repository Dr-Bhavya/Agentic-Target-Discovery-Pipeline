import os
import streamlit as st
import requests
import networkx as nx
import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq

# 1. STREAMLIT UI SETUP
st.set_page_config(page_title="Multi-Agent Systems Biology", layout="wide")
st.title("👥 TargetScout-AI: Multi-Agent Collaborative Target Discovery Panel")
st.caption("A free, local multi-agent system where specialized AI scientists collaborate on target validation.")

# 2. CONFIGURATION SIDEBAR
with st.sidebar:
    st.header("🔑 Configuration")
    user_api_key = st.text_input("Enter Free Groq API Key:", type="password")
    st.markdown("[Get a free Groq API key here](https://groq.com)")
    
    st.header("📊 Parameters")
    confidence_score = st.slider("STRING Interaction Confidence Cutoff", 400, 900, 400, step=100)

# 3. COMPUTATIONAL PIPELINE TOOLS

def build_string_network(gene_list_str: str) -> str:
    """Hits STRING-DB API and calculates NetworkX topological centralities to find hub genes."""
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    if not genes: return "Error: No valid gene symbols provided."
    
    url = "https://string-db.org"
    params = {"identifiers": "%0d".join(genes), "species": 9606, "required_score": confidence_score, "caller_identity": "targetscout"}
    
    try:
        response = requests.post(url, data=params)
        interactions = response.json()
        if not interactions: return "No network interactions found at this confidence cutoff."
        
        G = nx.Graph()
        for edge in interactions:
            G.add_edge(edge["preferredName_A"], edge["preferredName_B"], weight=edge["score"])
            
        df = pd.DataFrame([{
            "Gene": node,
            "Degree Centrality": round(nx.degree_centrality(G)[node], 4),
            "Betweenness (Bottleneck)": round(nx.betweenness_centrality(G)[node], 4),
            "Closeness (Proximity)": round(nx.closeness_centrality(G)[node], 4),
        } for node in G.nodes()])
        
        df["Consensus Rank Score"] = (df["Degree Centrality"] + df["Betweenness (Bottleneck)"] + df["Closeness (Proximity)"]) / 3
        df = df.sort_values(by="Consensus Rank Score", ascending=False).reset_index(drop=True)
        st.session_state["topology_df"] = df
        
        return f"Top 3 mathematically ranked consensus hub genes: {', '.join(df.head(3)['Gene'].tolist())}."
    except Exception as e:
        return f"Network tool failed: {str(e)}"

def run_functional_enrichment(gene_list_str: str) -> str:
    """Hits STRING Functional Enrichment API to find top GO and KEGG pathways."""
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    url = "https://string-db.org"
    try:
        res = requests.post(url, data={"identifiers": "%0d".join(genes), "species": 9606, "caller_identity": "targetscout"})
        results = res.json()
        if not results: return "No statistically significant functional pathways enriched."
        terms = [f"- [{t.get('category')}] {t.get('description')} (FDR: {t.get('fdr'):.4e})" for t in results[:5]]
        return "Top Enriched Biological Pathways:\n" + "\n".join(terms)
    except Exception as e:
        return f"Enrichment tool failed: {str(e)}"

def run_pubmed_literature_review(target_gene: str) -> str:
    """Queries public NCBI PubMed to check if target gene has therapeutic validation papers."""
    url = f"https://nih.gov{target_gene}[Title/Abstract]+AND+therapeutic+target&retmode=json&retmax=3"
    try:
        id_list = requests.get(url).json().get("esearchresult", {}).get("idlist", [])
        if not id_list: return f"No explicit target validation articles found for {target_gene}."
        return f"PubMed verification for {target_gene}: Found validation records. PMIDs: {', '.join(id_list)}."
    except Exception as e:
        return f"PubMed tracking failed: {str(e)}"


# 4. MULTI-AGENT ORCHESTRATION SETUP
default_genes = "SERPINE1, MMP1, MMP7, TGFB1, EGFR, STAT3, VEGFA"
input_genes = st.text_area("Provide a comma-separated list of Gene Symbols:", value=default_genes, height=100)

if st.button("🚀 Launch Multi-Agent Target Prioritization Panel"):
    if not user_api_key:
        st.error("Please add your free Groq API key in the sidebar configuration layout.")
    else:
        os.environ["GROQ_API_KEY"] = user_api_key
        free_model = Groq(id="llama-3.1-70b-versatile")
        
        # AGENT 1: The Network Topology Specialist
        network_analyst = Agent(
            name="Network Analyst Agent",
            role="Computes graph topology and maps protein-protein interactions",
            model=free_model,
            tools=[build_string_network],
            instructions=["Focus strictly on network structure. Calculate centralities and output the top hub genes."],
        )
        
        # AGENT 2: The Pathway & Functional Annotator
        pathway_specialist = Agent(
            name="Pathway Specialist Agent",
            role="Identifies biological mechanisms, GO terms, and KEGG pathways",
            model=free_model,
            tools=[run_functional_enrichment],
            instructions=["Focus strictly on biological pathways, ontology annotations, and false discovery rates."],
        )
        
        # AGENT 3: The Literature & Text Mining Expert
        literature_reviewer = Agent(
            name="Literature Reviewer Agent",
            role="Queries public medical databases for translational proof",
            model=free_model,
            tools=[run_pubmed_literature_review],
            instructions=["Focus strictly on text mining PubMed records to back up targets with published papers."],
        )
        
        # AGENT 4: The Master Supervisor Orchestrator
        orchestrator_agent = Agent(
            name="Discovery Team Orchestrator",
            model=free_model,
            team=[network_analyst, pathway_specialist, literature_reviewer],
            instructions=[
                "You are the Lead Scientific AI Orchestrator at Amgen.",
                "Delegate the input gene list to the Network Analyst Agent to find the top math hub genes.",
                "Delegate the input gene list to the Pathway Specialist Agent to get the active biological processes.",
                "Take the top hub gene identified by your Network Analyst and pass it to the Literature Reviewer Agent for a PubMed sweep.",
                "Synthesize inputs from all three agents into one master Executive Target Dossier. Clearly attribute which agent provided each insight.",
            ],
            markdown=True,
        )
        
        # 5. EXECUTION AND DISPLAY DISPLAY
        with st.status("🕵️‍♂️ Orchestrator managing multi-agent team communications...", expanded=True) as status:
            st.write("• Consulting Network Analyst to build mathematical graph matrix...")
            prompt_query = f"Coordinate a complete target validation pipeline for this gene list: {input_genes}"
            agent_response = orchestrator_agent.run(prompt_query)
            status.update(label="✅ Team Synthesis Complete!", state="complete")
            
        if "topology_df" in st.session_state:
            st.subheader("📊 Network Centrality Metrics (From Network Analyst)")
            st.dataframe(st.session_state["topology_df"], use_container_width=True)
            
        st.subheader("📋 Consolidated Master Target Dossier")
        st.markdown(agent_response.content)
