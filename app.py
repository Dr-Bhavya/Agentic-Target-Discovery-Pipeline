import os
import streamlit as st
import requests
import networkx as nx
import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq

# 1. STREAMLIT UI SETUP AND GRAPH MANAGEMENT
st.set_page_config(page_title="Multi-Agent Systems Biology", layout="wide")
st.title("👥 TargetScout-AI: Multi-Agent Collaborative Target Discovery Panel")
st.caption("A free, open-source multi-agent platform for systems biology discovery and target validation.")

# Initialize a cross-run session state for dataframe rendering
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

# 3. BACKGROUND COMPUTATIONAL PIPELINE TOOLS (GENE RETRIEVAL LOGIC)

def build_string_network(gene_list_str: str) -> str:
    """
    Connects programmatically to the STRING-DB REST API to fetch real interactions.
    Calculates Degree, Betweenness, and Closeness centralities to extract target hub genes.
    """
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    if not genes: 
        return "Error: No valid gene symbols were submitted."
    
    url = "https://string-db.org"
    params = {
        "identifiers": "%0d".join(genes), 
        "species": 9606, 
        "required_score": confidence_score, 
        "caller_identity": "targetscout_pipeline"
    }
    
    try:
        response = requests.post(url, data=params)
        interactions = response.json()
        if not interactions: 
            return "No network interactions found among these input genes at the chosen confidence cutoff."
        
        # Build network layout in-memory via NetworkX
        G = nx.Graph()
        for edge in interactions:
            p1 = edge["preferredName_A"]
            p2 = edge["preferredName_B"]
            score = edge["score"]
            G.add_edge(p1, p2, weight=score)
            
        # Calculate Math Centralities
        degree_dict = nx.degree_centrality(G)
        betweenness_dict = nx.betweenness_centrality(G)
        closeness_dict = nx.closeness_centrality(G)
        
        # Parse into organized metric rankings
        metrics = [{
            "Gene": node,
            "Degree Centrality": round(degree_dict[node], 4),
            "Betweenness (Bottleneck)": round(betweenness_dict[node], 4),
            "Closeness (Proximity)": round(closeness_dict[node], 4),
        } for node in G.nodes()]
        
        df = pd.DataFrame(metrics)
        df["Consensus Rank Score"] = (df["Degree Centrality"] + df["Betweenness (Bottleneck)"] + df["Closeness (Proximity)"]) / 3
        df = df.sort_values(by="Consensus Rank Score", ascending=False).reset_index(drop=True)
        
        # Commit dataframe securely to streamlit state for rendering
        st.session_state["topology_df"] = df
        
        top_hubs = df.head(3)["Gene"].tolist()
        return f"Network calculation finished. Analyzed {len(G.nodes())} interacting nodes. The top 3 mathematically ranked consensus hub genes are: {', '.join(top_hubs)}."
    except Exception as e:
        return f"Network Analyst topological processing failed: {str(e)}"

def run_functional_enrichment(gene_list_str: str) -> str:
    """
    Queries the public STRING Functional Enrichment API to track functional annotations.
    """
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    url = "https://string-db.org"
    params = {
        "identifiers": "%0d".join(genes), 
        "species": 9606, 
        "caller_identity": "targetscout_pipeline"
    }
    try:
        res = requests.post(url, data=params)
        results = res.json()
        if not results: 
            return "No statistically significant functional pathways or GO terms enriched for this set."
            
        terms = [f"- [{t.get('category')}] {t.get('description')} (FDR: {t.get('fdr'):.4e})" for t in results[:5]]
        return "Top Enriched Biological Pathways and Gene Ontologies:\n" + "\n".join(terms)
    except Exception as e:
        return f"Pathway Specialist enrichment lookup failed: {str(e)}"

def run_pubmed_literature_review(target_gene: str) -> str:
    """
    Queries NCBI PubMed via live E-Utilities to extract publication proof for target therapeutic relevance.
    """
    gene_clean = target_gene.upper().strip()
    url = f"https://nih.gov{gene_clean}[Title/Abstract]+AND+therapeutic+target&retmode=json&retmax=3"
    try:
        res = requests.get(url).json()
        id_list = res.get("esearchresult", {}).get("idlist", [])
        if not id_list: 
            return f"PubMed search completed: No explicit therapeutic target validation records located for gene {gene_clean}."
        return f"PubMed Literature Sweep for {gene_clean}: Identified valid target evidence publications. Associated PMIDs: {', '.join(id_list)}."
    except Exception as e:
        return f"Literature Reviewer retrieval error: {str(e)}"


# 4. FRONT-END INTERFACE LAYOUT
default_genes = "SERPINE1, MMP1, MMP7, TGFB1, EGFR, STAT3, VEGFA"
input_genes = st.text_area("Provide a comma-separated list of Gene Symbols:", value=default_genes, height=100)

if st.button("🚀 Launch Multi-Agent Target Prioritization Panel"):
    if not user_api_key:
        st.error("Please add your free Groq API key in the sidebar configuration layout.")
    else:
        # Pass key safely into local environment variables
        os.environ["GROQ_API_KEY"] = user_api_key
        
        # Define the stable production model ID across agents
        production_model = Groq(id="llama-3.3-70b-versatile")
        
        # AGENT 1: The Network Topology Specialist
        network_analyst = Agent(
            name="Network Analyst Agent",
            role="Computes graph topology and maps protein-protein interactions",
            model=production_model,
            tools=[build_string_network],
            instructions=["Focus strictly on network structure. Calculate centralities and output the top hub genes. Avoid long explanations."],
        )
        
        # AGENT 2: The Pathway & Functional Annotator
        pathway_specialist = Agent(
            name="Pathway Specialist Agent",
            role="Identifies biological mechanisms, GO terms, and KEGG pathways",
            model=production_model,
            tools=[run_functional_enrichment],
            instructions=["Focus strictly on biological pathways, ontology annotations, and false discovery rates. Provide a direct, factual summary."],
        )
        
        # AGENT 3: The Literature & Text Mining Expert
        literature_reviewer = Agent(
            name="Literature Reviewer Agent",
            role="Queries public medical databases for translational proof",
            model=production_model,
            tools=[run_pubmed_literature_review],
            instructions=["Focus strictly on text mining PubMed records to back up targets with published papers. Return PMIDs clearly."],
        )
        
        # AGENT 4: The Master Supervisor Orchestrator
        orchestrator_agent = Agent(
            name="Discovery Team Orchestrator",
            model=production_model,
            team=[network_analyst, pathway_specialist, literature_reviewer],
            instructions=[
                "You are the Lead Scientific AI Orchestrator at Amgen.",
                "First, delegate the input gene list to the Network Analyst Agent to calculate the highest math hub genes.",
                "Second, delegate the input gene list to the Pathway Specialist Agent to pull out active functional mechanisms.",
                "Third, take the highest-scoring single gene extracted from the Network Analyst's report and pass it to the Literature Reviewer Agent for a PubMed search.",
                "Compile all outputs into a single, cohesive, multi-section Executive Target Prioritization Dossier. Clearly attribute which agent provided each insight.",
            ],
            markdown=True,
        )
        
        # 5. EXECUTION FEEDBACK CONTAINER AND OUTPUT RENDERING
        with st.status("🕵️‍♂️ Orchestrator managing multi-agent team communications...", expanded=True) as status:
            st.write("• Consulting Network Analyst to parse biological graph data lines...")
            
            prompt_query = f"Coordinate a complete target validation pipeline for this gene list: {input_genes}"
            agent_response = orchestrator_agent.run(prompt_query)
            
            status.update(label="✅ Team Synthesis Complete!", state="complete")
            
        # Display the mathematical metric grid if processed successfully
        if st.session_state["topology_df"] is not None:
            st.subheader("📊 Network Centrality Metrics (From Network Analyst)")
            st.dataframe(st.session_state["topology_df"], use_container_width=True)
            
        # Display the formatted multi-agent report dossier 
        st.subheader("📋 Consolidated Master Target Dossier")
        st.markdown(agent_response.content)
