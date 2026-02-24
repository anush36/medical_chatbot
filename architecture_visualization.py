import graphviz

dot = graphviz.Digraph(
    'System Architecture',
    filename='system_architecture',
    format='png',
    graph_attr={
        'rankdir': 'TB',
        'nodesep': '0.8',
        'ranksep': '0.9',
        'splines': 'polyline', 
        'compound': 'true' 
    }
)

# Global styling 
dot.attr('node', shape='box', style='rounded,filled', fontsize='24', fontname='Helvetica', margin='0.3,0.2', penwidth='2')
dot.attr('edge', fontsize='18', fontname='Helvetica', color='#555555', penwidth='2')

dot.node('User', 'End Users', shape='ellipse', fillcolor='#f0f0f0')

# Invisible anchor to force the 3-column side-by-side layout
dot.node('Anchor', style='invis', shape='point', width='0', height='0')

# --- COLUMN 1: Client & API Layer ---
with dot.subgraph(name='cluster_client') as c:
    c.attr(label='Client & API Layer', style='dashed', color='gray', fontsize='20')
    c.node('UI', 'Streamlit UI\n(app.py)', fillcolor='#e1f5fe')
    c.node('API', 'FastAPI Server\n(main.py)', fillcolor='#f3e5f5')
    
    c.edge('UI', 'API', label=' HTTP POST')

# --- COLUMN 2: LangGraph Workflow ---
with dot.subgraph(name='cluster_workflow') as c:
    c.attr(label='LangGraph Workflow', style='dashed', color='black', fontsize='20')
    
    c.node('Draft', 'Draft Node', fillcolor='#e8f5e9')
    c.node('Safety', 'Safety Node', fillcolor='#e8f5e9')
    c.node('Extract', 'Extract Claims', fillcolor='#e8f5e9')
    c.node('Retrieve', 'Retrieve Node', fillcolor='#e8f5e9')
    c.node('Verify', 'Verify Node', fillcolor='#e8f5e9')
    
    # Internal DAG Logic
    c.edge('Draft', 'Safety')
    c.edge('Safety', 'Extract', label=' If Safe')
    
    # Padded the labels so they don't hug the lines too tightly
    c.edge('Safety', 'Draft', constraint='false', tailport='e', headport='e', label='  If Unsafe  ', fontcolor='#b71c1c', color='#b71c1c')
    
    c.edge('Extract', 'Retrieve')
    c.edge('Retrieve', 'Verify')
    
    # Padded the labels so they don't hug the lines too tightly
    c.edge('Verify', 'Draft', constraint='false', tailport='e', headport='e', label='  If Unvalidated  ', fontcolor='#b71c1c', color='#b71c1c')

# --- COLUMN 3: GCP Infrastructure ---
with dot.subgraph(name='cluster_infra') as c:
    c.attr(label='GCP Infrastructure', style='dashed', color='gray', fontsize='20')
    c.node('CloudRun', 'Cloud Run Container\n(vLLM Server)', fillcolor='#fff3e0')
    c.node('GPU', 'NVIDIA L4 GPU\n(MedGemma-4b-it)', fillcolor='#fff3e0')
    
    c.edge('CloudRun', 'GPU')

# --- COLUMN 3 (Bottom): Knowledge & RAG ---
with dot.subgraph(name='cluster_knowledge') as c:
    c.attr(label='Knowledge Retrieval', style='dashed', color='gray', fontsize='20')
    c.node('VectorDB', 'Vector DB\n(RAG Embeddings)', shape='cylinder', fillcolor='#e0f7fa') # Light cyan cylinder
    c.node('PMC', 'PubMed Central API', shape='box', fillcolor='#fff9c4')
    
    c.edge('VectorDB', 'PMC', style='invis') # Keep them stacked neatly

# --- Layout Enforcement ---
dot.edge('User', 'UI')
dot.edge('Anchor', 'UI', style='invis')
dot.edge('Anchor', 'Draft', style='invis')
dot.edge('Anchor', 'CloudRun', style='invis')
dot.edge('GPU', 'VectorDB', style='invis') # Push the Knowledge cluster under the GPU

# --- Cross-Column Connections ---
# 1. API to Workflow Block (Executes Workflow text removed)
dot.edge('API', 'Draft', lhead='cluster_workflow', constraint='false')

# 2. Workflow Block back to API (Now says "If Grounded")
dot.edge('Verify', 'API', ltail='cluster_workflow', constraint='false', label=' If Grounded', color='#1b5e20', fontcolor='#1b5e20')

# 3. Clean Inference API call
dot.edge('Safety', 'CloudRun', ltail='cluster_workflow', constraint='false', style='dashed', label=' LLM Inference API', color='#9e9e9e', fontcolor='#757575')

# 4. Search Literature & RAG (dashed and greyed out)
dot.edge('Retrieve', 'VectorDB', constraint='false', label=' RAG Query', style='dashed', color='#9e9e9e', fontcolor='#757575')
dot.edge('Retrieve', 'PMC', constraint='false', label=' Search Literature', style='dashed', color='#9e9e9e', fontcolor='#757575')

# Render the output image
print("Attempting to render image...")
dot.render(cleanup=True)
print("Success! Created system_architecture.png")