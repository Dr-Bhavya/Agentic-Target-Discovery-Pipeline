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
    st.markdown("[Get a free Groq API key here](https://groq.com)")
    
    st.header("📊 Parameters")
    confidence_score = st.slider("STRING Interaction Confidence Cutoff", 400, 900, 400, step=100)
    st.caption("400 = Medium Confidence, 700 = High Confidence")
    
    # Critical Fix Parameter: Pull interactors if the input genes are not linked directly
    add_nodes = st.number_input("Add Interactors (Neighborhood Expansion)", min_value=0, max_value=20, value=5)

# 3. CONTEXT-SAFE BIOINFORMATICS PIPELINES

def run_network_topology_pipeline(gene_list_str: str) -> dict:
    """
    Connects to the STRING-DB interaction endpoint.
    Expands the network neighborhood if direct connections are lacking.
    """
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    if not genes: 
        return {"status": "error", "message": "No valid gene symbols provided."}
    
    # Use version-specific production endpoint for high stability
    url = "https://string-db.org"
    
    params = {
        "identifiers": "\n".join(genes), 
        "species": 9606,  # Human
        "required_score": confidence_score, 
        "add_nodes": add_nodes,  # Crucial fix to prevent empty network error sheets
        "caller_identity": "targetscout_v3"
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            return {"status": "error", "message": f"STRING-DB Server error code {response.status_code}."}
            
        try:
            interactions = response.json()
        except ValueError:
            return {"status": "error", "message": "STRING-DB responded with text data layout. Try adding more interactors in the sidebar."}
            
        if not interactions or not isinstance(interactions, list): 
            return {"status": "error", "message": "No network interaction partners found for these genes."}
        
        # Construct graph structures
        G = nx.Graph()
        for edge in interactions:
            p1 = edge.get("preferredName_A")
            p2 = edge.get("preferredName_B")
            score = edge.get("score")
            if p1 and p2:
                G.add_edge(p1, p2, weight=score)
                
        if G.number_of_nodes() == 0:
            return {"status": "error", "message": "Graph generation failed. zero interactions mapped."}
            
        # Compute centralities
        deg_cent = nx.degree_centrality(G)
        bet_cent = nx.betweenness_centrality(G) if len(G.nodes()) > 2 else {n: 0.0 for n in G.nodes()}
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
        
        # Explicitly pull the gene string using standard positional index (.iloc[0]["Gene"])
        top_gene_name = str(df.iloc[0]["Gene"]) if not df.empty else genes[0]
        
        return {
            "status": "success",
            "top_genes": top_gene_name,
            "raw_text": f"Successfully parsed {len(G.nodes())} biological interaction nodes. Top mathematically prioritized target hub is: {top_gene_name}."
        }
    except Exception as e:
        return {"status": "error", "message": f"Network layout script error: {str(e)}"}

def run_functional_enrichment_pipeline(gene_list_str: str) -> str:
    """Fetches enrichment pathways with secure handling for GET formatting."""
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    url = "https://string-db.org"
    try:
        res = requests.get(url, params={"identifiers": "\n".join(genes), "species": 9606, "caller_identity": "targetscout_v3"})
        if res.status_code != 200:
            return "Could not compute enrichment profiles due to downstream server errors."
        results = res.json()
        if not results or not isinstance(results, list): 
            return "No statistically significant GO/KEGG pathway terms matched."
            
        terms = [f"- [{t.get('category')}] {t.get('description')} (FDR: {t.get('fdr'):.4e})" for t in results[:5]]
        return "Top Enriched Functional Processes and Pathways:\n" + "\n".join(terms)
    except Exception as e:
        return f"Enrichment pipeline parsing down: {str(e)}"

def run_pubmed_literature_pipeline(target_gene: str) -> str:
    """Queries PubMed API using standard URL token encoders."""
    gene = target_gene.upper().strip()
    url = f"https://nih.gov{gene}[Title/Abstract]+AND+therapeutic+target&retmode=json&retmax=3"
    try:
        res = requests.get(url).json()
        id_list = res.get("esearchresult", {}).get("idlist", [])
        if not id_list: 
            return f"No baseline translational tracking records found for target molecule {gene} in active medical publications."
        return f"PubMed Verification Search for {gene}: Identified supportive target validation articles. Associated PMIDs: {', '.join(id_list)}."
    except Exception as e:
        return f"Literature data sweep dropped: {str(e)}"


# 4. FRONT-END PANEL LOGIC
default_genes = "SERPINE1, MMP1, MMP7, TGFB1, EGFR, STAT3, VEGFA"
input_genes = st.text_area("Provide a comma-separated list of Gene Symbols:", value=default_genes, height=100)

if st.button("🚀 Launch Autonomous Target Prioritization Pipeline"):
    if not user_api_key:
        st.error("Please add your free Groq API key in the sidebar configuration layout.")
    else:
        os.environ["GROQ_API_KEY"] = user_api_key
        
        with st.status("🕵️‍♂️ Executing multi-stage systems biology workflow...", expanded=True) as status:
            
            # Stage 1: Run Network Mapping
            st.write("1. Initializing Network Analyst Layer → Querying STRING-DB matrix & running centralities...")
            net_results = run_network_topology_pipeline(input_genes)
            
            if net_results["status"] == "error":
                st.error(net_results["message"])
                status.update(label="❌ Pipeline aborted due to data parsing error", state="error")
                st.stop()
                
            network_context = net_results["raw_text"]
            top_candidate = net_results["top_genes"]
            
            # Stage 2: Functional Annotation
            st.write("2. Initializing Pathway Specialist Layer → Constructing functional ontology mappings...")
            enrichment_context = run_functional_enrichment_pipeline(input_genes)
            
            # Stage 3: Text Mining RAG
            st.write(f"3. Initializing Literature Reviewer Layer → Mining live clinical evidence for top hub: {top_candidate}...")
            pubmed_context = run_pubmed_literature_pipeline(top_candidate)
            
            # Stage 4: AI Synthesis Generation
            st.write("4. Directing all extracted knowledge streams to AI Synthesis Lead for target reporting...")
            
            target_evaluation_agent = Agent(
                name="Amgen-Style Target Discovery Lead",
                model=Groq(id="llama-3.3-70b-versatile"),
                instructions=[
                    "You are an expert GCF6 Agentic AI Lead specializing in Target Discovery at Amgen.",
                    "Review the analytical data payloads below, analyze the network values, and build an executive prioritized candidate report.",
                    "Structure your output cleanly with dedicated titles for: 1. Graph Structural Insights, 2. Pathway Mapping, and 3. Clinical Tractability Recommendations.",
                ],
                markdown=True,
            )
            
            augmented_prompt = f"""
            Synthesize this compiled systems biology evidence into an Executive Target Dossier:
            
            [USER LIST]: {input_genes}
            [NETWORK INTERACTORS EXTRAPOLATION]: {network_context}
            [FUNCTIONAL ONTOLOGY ANNOTATIONS]: {enrichment_context}
            [PUBMED TARGET VALIDATION VERIFICATION]: {pubmed_context}
            """
            
            agent_response = target_evaluation_agent.run(augmented_prompt)
            status.update(label="✅ Discovery Pipeline Synthesis Complete!", state="complete")
            
        # 5. RENDER OUTPUT TABLES AND EVALUATIONS
        if st.session_state["topology_df"] is not None:
            st.subheader("📊 Network Centrality Metrics (Computed via NetworkX)")
            st.dataframe(st.session_state["topology_df"], use_container_width=True)
            
        st.subheader("📋 Consolidated Master Target Dossier")
        st.markdown(agent_response.content)
