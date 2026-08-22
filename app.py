import os
import streamlit as st
import requests
import networkx as nx
import pandas as pd
from phi.agent import Agent
from phi.model.groq import Groq

# 1. STREAMLIT UI SETUP AND CONFIGURATION
st.set_page_config(page_title="Agentic Target Prioritization", layout="wide")
st.title("🧬 TargetScout-AI: Systems Biology Target Prioritization Pipeline")
st.caption("A stable, high-utility agentic platform combining real-time STRING-DB data, NetworkX topology, and live PubMed RAG.")

if "topology_df" not in st.session_state:
    st.session_state["topology_df"] = None

# 2. CONFIGURATION SIDEBAR LAYOUT
with st.sidebar:
    st.header("🔑 Configuration")
    user_api_key = st.text_input("Enter Free Groq API Key:", type="password")
    st.markdown("[Get a free Groq API key here](https://groq.com)")
    
    st.header("📊 Parameters")
    confidence_score = st.slider("STRING Interaction Confidence Cutoff", 400, 900, 400, step=100)
    st.caption("400 = Medium Confidence, 700 = High Confidence")
    
    add_nodes = st.number_input("Add Interactors (Neighborhood Expansion)", min_value=0, max_value=20, value=5)

# 3. ROBUST BIOINFORMATICS COMPUTATIONAL CHUNKS

def run_network_topology_pipeline(gene_list_str: str) -> dict:
    """
    Connects to the STRING-DB interaction network endpoint via standard POST parameters.
    Extracts topological centralities using NetworkX math.
    """
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    if not genes: 
        return {"status": "error", "message": "No valid gene symbols provided."}
    
    # FIX: Pointed to the actual JSON endpoint route instead of the website homepage
    url = "https://string-db.org"
    payload = {
        "identifiers": "\n".join(genes), 
        "species": 9606,  # Homo Sapiens
        "required_score": confidence_score, 
        "add_nodes": int(add_nodes),
        "caller_identity": "targetscout_final"
    }
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            return {"status": "error", "message": f"STRING-DB Server error code {response.status_code}."}
            
        try:
            interactions = response.json()
        except ValueError:
            return {"status": "error", "message": "STRING-DB returned text format layout instead of structured JSON parameters."}
            
        if not interactions or (isinstance(interactions, dict) and "message" in interactions): 
            return {"status": "error", "message": "No network interaction partners matched for these genes."}
        
        # Assemble Graph Matrix
        G = nx.Graph()
        for edge in interactions:
            p1 = edge.get("preferredName_A")
            p2 = edge.get("preferredName_B")
            score = edge.get("score")
            if p1 and p2:
                G.add_edge(p1, p2, weight=score)
                
        if G.number_of_nodes() == 0:
            return {"status": "error", "message": "Graph generation failed. Matrix nodes are zero."}
            
        # Compute exact topological vectors
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
        
        top_gene_name = str(df.iloc[0]["Gene"]) if not df.empty else genes[0]
        
        return {
            "status": "success",
            "top_genes": top_gene_name,
            "raw_text": f"Parsed {len(G.nodes())} interactive graph network items. Top prioritized biological candidate hub gene is: {top_gene_name}."
        }
    except Exception as e:
        return {"status": "error", "message": f"Network analysis pipeline down: {str(e)}"}

def run_functional_enrichment_pipeline(gene_list_str: str) -> str:
    """Fetches GO process and KEGG functional categories from STRING database matrix models."""
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    # FIX: Pointed to the actual enrichment JSON endpoint route
    url = "https://string-db.org"
    try:
        res = requests.post(url, data={"identifiers": "\n".join(genes), "species": 9606, "caller_identity": "targetscout_final"})
        if res.status_code != 200:
            return "Could not compute functional ontology sets."
        results = res.json()
        if not results or not isinstance(results, list): 
            return "No statistically significant pathway items matched for the input list."
            
        terms = [f"- [{t.get('category')}] {t.get('description')} (FDR: {t.get('fdr'):.4e})" for t in results[:5]]
        return "Top Enriched Pathway Alignments:\n" + "\n".join(terms)
    except Exception as e:
        return f"Enrichment pipeline parsing exception error: {str(e)}"

def run_pubmed_literature_pipeline(target_gene: str) -> str:
    """Performs real-time abstract scraping via public NCBI E-Utilities token parameters."""
    gene = target_gene.upper().strip()
    # FIX: Corrected malformed NCBI Eutils URL construction
    url = f"https://nih.gov{gene}[Title/Abstract]+AND+therapeutic+target&retmode=json&retmax=3"
    try:
        res = requests.get(url).json()
        id_list = res.get("esearchresult", {}).get("idlist", [])
        if not id_list: 
            return f"No baseline clinical validation publications discovered explicitly naming target {gene} on PubMed database."
        return f"PubMed Verification Search for {gene}: Located target research proof. Associated PMIDs: {', '.join(id_list)}."
    except Exception as e:
        return f"PubMed literature verification pipeline disconnected: {str(e)}"


# 4. STREAMLIT APPLICATION CONTROLLER PANEL
default_genes = "SERPINE1, MMP1, MMP7, TGFB1, EGFR, STAT3, VEGFA"
input_genes = st.text_area("Provide a comma-separated list of Gene Symbols:", value=default_genes, height=100)

if st.button("🚀 Launch Autonomous Target Prioritization Pipeline"):
    if not user_api_key:
        st.error("Please add your free Groq API key in the sidebar configuration layout.")
    else:
        os.environ["GROQ_API_KEY"] = user_api_key
        
        with st.status("🕵️‍♂️ Executing multi-stage systems biology workflow...", expanded=True) as status:
            
            # Stage 1: Network Construction & Metrics
            st.write("1. Initializing Network Analyst Layer → Querying STRING-DB matrix & running centralities...")
            net_results = run_network_topology_pipeline(input_genes)
            
            if net_results["status"] == "error":
                st.error(net_results["message"])
                status.update(label="❌ Pipeline aborted due to data parsing error", state="error")
                st.stop()
                
            network_context = net_results["raw_text"]
            top_candidate = net_results["top_genes"]
            
            # Stage 2: Functional Annotation Tracking
            st.write("2. Initializing Pathway Specialist Layer → Constructing functional ontology mappings...")
            enrichment_context = run_functional_enrichment_pipeline(input_genes)
            
            # Stage 3: Real-Time PubMed Literature Sweep
            st.write(f"3. Initializing Literature Reviewer Layer → Mining live clinical evidence for top hub: {top_candidate}...")
            pubmed_context = run_pubmed_literature_pipeline(top_candidate)
            
            # Stage 4: Dynamic RAG Integration & Inference
            st.write("4. Directing all extracted knowledge streams to AI Synthesis Lead for target reporting...")
            
            # FIX: Completed the truncated agent code block and stream generation block
            target_evaluation_agent = Agent(
                name="Amgen-Style Target Discovery Lead",
                model=Groq(id="llama3-8b-8192"),
                instructions=[
                    "You are a Senior Systems Biology & Target Discovery Scientist.",
                    "Analyze the provided network topology metrics, functional pathway context, and PubMed clinical validation context.",
                    "Synthesize a therapeutic target brief detailing why this gene acts as a viable druggable hub bottleneck candidate."
                ],
                markdown=True
            )
            
            combined_prompt = f"""
            Synthesize a target report based on these real-time pipelines:
            
            [NETWORK METRICS]: {network_context}
            [FUNCTIONAL ONTOLOGY]: {enrichment_context}
            [LITERATURE EVIDENCE]: {pubmed_context}
            """
            
            agent_response = target_evaluation_agent.run(combined_prompt)
            status.update(label="✨ Pipeline completed successfully!", state="complete")
            
        # Display Results Panels
        st.subheader("📊 Topological Prioritization Matrix")
        if st.session_state["topology_df"] is not pd.DataFrame:
            st.dataframe(st.session_state["topology_df"], use_container_width=True)
            
        st.subheader("🔬 AI Target Discovery Evaluation Report")
        st.markdown(agent_response.content)
