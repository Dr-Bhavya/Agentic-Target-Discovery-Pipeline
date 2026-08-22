import os
import streamlit as st
import requests
import networkx as nx
import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq

# 1. STREAMLIT UI SETUP AND PERFORMANCE MANAGEMENT
st.set_page_config(page_title="Agentic Target Prioritization", layout="wide")
st.title("🧬 TargetScout-AI: Systems Biology Target Prioritization Pipeline")
st.caption("A stable, high-utility agentic platform combining real-time STRING-DB data, NetworkX topology, and live PubMed RAG.")

if "topology_df" not in st.session_state:
    st.session_state["topology_df"] = None

# 2. CONFIGURATION SIDEBAR
with st.sidebar:
    st.header("🔑 Configuration")
    user_api_key = st.text_input("Enter Free Groq API Key:", type="password")
    st.markdown("[Get a free Groq API key here](https://console.groq.com/)")
    
    st.header("📊 Parameters")
    confidence_score = st.slider("STRING Interaction Confidence Cutoff", 400, 900, 400, step=100)
    st.caption("400 = Medium Confidence, 700 = High Confidence")

# 3. HIGH-UTILITY BIOINFORMATICS TOOLS (ROBUST EXECUTION)

def run_network_topology_pipeline(gene_list_str: str) -> dict:
    """
    Acts as the Network Analyst tool. Connects to STRING-DB API,
    builds a network graph via NetworkX, and computes mathematical centralities.
    """
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    if not genes: 
        return {"status": "error", "message": "No valid gene symbols provided."}
    
    url = "https://string-db.org"
    params = {
        "identifiers": "%0d".join(genes), 
        "species": 9606, 
        "required_score": confidence_score, 
        "caller_identity": "targetscout"
    }
    
    try:
        response = requests.post(url, data=params)
        interactions = response.json()
        if not interactions or "message" in interactions: 
            return {"status": "error", "message": "No functional protein network interactions found."}
        
        # Build NetworkX structural model
        G = nx.Graph()
        for edge in interactions:
            G.add_edge(edge["preferredName_A"], edge["preferredName_B"], weight=edge["score"])
            
        # Compute discrete mathematical centralities
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G)
        clo_cent = nx.closeness_centrality(G)
        
        metrics = [{
            "Gene": node,
            "Degree Centrality": round(deg_cent[node], 4),
            "Betweenness (Bottleneck)": round(bet_cent[node], 4),
            "Closeness (Proximity)": round(clo_cent[node], 4),
        } for node in G.nodes()]
        
        df = pd.DataFrame(metrics)
        df["Consensus Rank Score"] = (df["Degree Centrality"] + df["Betweenness (Bottleneck)"] + df["Closeness (Proximity)"]) / 3
        df = df.sort_values(by="Consensus Rank Score", ascending=False).reset_index(drop=True)
        
        st.session_state["topology_df"] = df
        return {
            "status": "success",
            "top_genes": df.head(3)["Gene"].tolist(),
            "raw_text": f"Analyzed {len(G.nodes())} interactive network nodes. Top 3 highest consensus hub nodes: {', '.join(df.head(3)['Gene'].tolist())}."
        }
    except Exception as e:
        return {"status": "error", "message": f"Network analysis pipeline down: {str(e)}"}

def run_functional_enrichment_pipeline(gene_list_str: str) -> str:
    """
    Acts as the Pathway Specialist tool. Fetches statistically significant
    GO terms and KEGG pathways from the STRING enrichment endpoint.
    """
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    url = "https://string-db.org"
    try:
        res = requests.post(url, data={"identifiers": "%0d".join(genes), "species": 9606, "caller_identity": "targetscout"})
        results = res.json()
        if not results or "message" in results: 
            return "No statistically significant functional pathway enrichments located."
            
        terms = [f"- [{t.get('category')}] {t.get('description')} (FDR: {t.get('fdr'):.4e})" for t in results[:5]]
        return "Top Enriched Functional Processes and Pathways:\n" + "\n".join(terms)
    except Exception as e:
        return f"Functional enrichment calculation failed: {str(e)}"

def run_pubmed_literature_pipeline(target_gene: str) -> str:
    """
    Acts as the Literature Reviewer tool. Performs live text mining RAG 
    by querying NCBI PubMed via E-Utilities for target clinical relevance.
    """
    gene = target_gene.upper().strip()
    url = f"https://nih.gov{gene}[Title/Abstract]+AND+therapeutic+target&retmode=json&retmax=3"
    try:
        res = requests.get(url).json()
        id_list = res.get("esearchresult", {}).get("idlist", [])
        if not id_list: 
            return f"No historical clinical validation papers found explicitly linking {gene} as a therapeutic target on PubMed."
        return f"PubMed Search for {gene}: Located supportive peer-reviewed research. Associated PMIDs: {', '.join(id_list)}."
    except Exception as e:
        return f"PubMed literature tracking failed: {str(e)}"


# 4. COMPACT FRONT-END WORKFLOW INTERFACE
default_genes = "SERPINE1, MMP1, MMP7, TGFB1, EGFR, STAT3, VEGFA"
input_genes = st.text_area("Provide a comma-separated list of Gene Symbols:", value=default_genes, height=100)

if st.button("🚀 Launch Autonomous Target Prioritization Pipeline"):
    if not user_api_key:
        st.error("Please add your free Groq API key in the sidebar configuration layout.")
    else:
        os.environ["GROQ_API_KEY"] = user_api_key
        
        with st.status("🕵️‍♂️ Executing multi-stage systems biology workflow...", expanded=True) as status:
            
            # Stage 1: Run Network Math Programmatically
            st.write("1. Initializing Network Analyst Layer → Fetching STRING-DB interactions & computing centralities...")
            net_results = run_network_topology_pipeline(input_genes)
            
            if net_results["status"] == "error":
                st.error(net_results["message"])
                status.update(label="❌ Pipeline aborted due to network error", state="error")
                st.stop()
                
            network_context = net_results["raw_text"]
            top_candidate = net_results["top_genes"][0]
            
            # Stage 2: Run Enrichment Programmatically
            st.write("2. Initializing Pathway Specialist Layer → Calculating functional ontology annotations...")
            enrichment_context = run_functional_enrichment_pipeline(input_genes)
            
            # Stage 3: Run Literature Review on Top Math Hub Gene Programmatically
            st.write(f"3. Initializing Literature Reviewer Layer → Extracting PubMed verification for top hub gene: {top_candidate}...")
            pubmed_context = run_pubmed_literature_pipeline(top_candidate)
            
            # Stage 4: Feed Compiled Data to the AI Synthesis Lead
            st.write("4. Directing full contextual streams to AI Synthesis Lead for final target evaluation reporting...")
            
            target_evaluation_agent = Agent(
                name="Amgen-Style Target Discovery Lead",
                model=Groq(id="llama-3.3-70b-versatile"),
                instructions=[
                    "You are an expert GCF6 Agentic AI Lead specializing in Disease Biology and Target Discovery at Amgen.",
                    "Your task is to take the provided multi-source data streams and synthesize them into a professional executive dossier.",
                    "Ensure your output includes dedicated sections for: 1. Network Topology Insights, 2. Functional & Pathway Annotations, and 3. Translational Clinical Recommendations.",
                    "Clearly state why the top candidate gene was selected based on the mathematical centrality metrics provided.",
                ],
                markdown=True,
            )
            
            # Construct the comprehensive, augmented prompt packet (RAG Mechanics)
            augmented_prompt = f"""
            Please review and synthesize the following data payload into an Executive Target Prioritization Dossier:
            
            [USER INPUT GENES]: {input_genes}
            
            [RETRIEVED NETWORK ANALYSIS DATA]:
            {network_context}
            
            [RETRIEVED FUNCTIONAL ENRICHMENT DATA]:
            {enrichment_context}
            
            [RETRIEVED PUBMED LITERATURE DATA FOR TOP TARGET]:
            {pubmed_context}
            """
            
            # Run inference cleanly using the verified production model ID
            agent_response = target_evaluation_agent.run(augmented_prompt)
            status.update(label="✅ Discovery Pipeline Synthesis Complete!", state="complete")
            
        # 5. RENDER DERIVED ENGINE METRICS AND DOSSIER OUTPUT
        if st.session_state["topology_df"] is not None:
            st.subheader("📊 Network Centrality Metrics (Computed via NetworkX)")
            st.dataframe(st.session_state["topology_df"], use_container_width=True)
            
        st.subheader("📋 Consolidated Master Target Dossier")
        st.markdown(agent_response.content)
