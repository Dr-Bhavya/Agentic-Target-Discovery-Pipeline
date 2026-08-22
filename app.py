def run_network_topology_pipeline(gene_list_str: str) -> dict:
    """
    Connects to the STRING-DB interaction network endpoint via standard POST parameters.
    Extracts topological centralities using NetworkX math.
    """
    genes = [g.strip().upper() for g in gene_list_str.split(",") if g.strip()]
    if not genes: 
        return {"status": "error", "message": "No valid gene symbols provided."}
    
    url = "https://string-db.org/api/json/network"
    payload = {
        "identifiers": "\n".join(genes), 
        "species": 9606,  # Homo Sapiens
        "required_score": confidence_score, 
        # FIX 1: Parameter name must be 'add_white_nodes' per official STRING API spec
        "add_white_nodes": int(add_nodes),
        "caller_identity": "targetscout_final"
    }
    
    try:
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            return {"status": "error", "message": f"STRING-DB Server error code {response.status_code}: {response.text[:200]}"}
            
        try:
            interactions = response.json()
        except ValueError:
            return {
                "status": "error", 
                "message": f"STRING-DB returned plain text instead of JSON. Server Response: {response.text[:250]}"
            }
            
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
        
        # FIX 2: Corrected .iloc syntax to isolate the top-ranked gene name string
        top_gene_name = str(df.iloc[0]["Gene"]) if not df.empty else genes[0]
        
        return {
            "status": "success",
            "top_genes": top_gene_name,
            "raw_text": f"Parsed {len(G.nodes())} interactive graph network items. Top prioritized biological candidate hub gene is: {top_gene_name}."
        }
    except Exception as e:
        return {"status": "error", "message": f"Network analysis pipeline down: {str(e)}"}
