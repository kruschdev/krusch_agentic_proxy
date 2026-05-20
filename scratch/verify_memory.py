import psycopg2
import sys

def query_memory():
    try:
        conn = psycopg2.connect(
            host="10.0.0.85",
            port="5434",
            database="kruschdb",
            user="openclaw",
            password="openclaw_password"
        )
        cur = conn.cursor()
        
        print("--- Querying homelab_memory_v2 ---")
        cur.execute("""
            SELECT category, content, project, created_at 
            FROM homelab_memory_v2 
            WHERE content ILIKE '%krusch-agentic-proxy%' 
               OR content ILIKE '%krusch-agentic-mcp%' 
               OR project ILIKE '%krusch-agentic-proxy%' 
               OR project ILIKE '%krusch-agentic-mcp%' 
               OR content ILIKE '%gpu%'
            ORDER BY created_at DESC LIMIT 10;
        """)
        rows = cur.fetchall()
        for r in rows:
            print(f"[{r[3]}] Category: {r[0]} | Project: {r[2]}")
            print(f"Content: {r[1]}")
            print("-" * 50)
            
        print("\n--- Querying ide_agent_nuggets (Steering Facts/AMD GPU) ---")
        cur.execute("""
            SELECT key, value, kind, project, created_at 
            FROM ide_agent_nuggets 
            WHERE key ILIKE '%gpu%' 
               OR value ILIKE '%gpu%' 
               OR key ILIKE '%krusch-agentic-proxy%' 
               OR value ILIKE '%krusch-agentic-proxy%'
            ORDER BY created_at DESC LIMIT 10;
        """)
        rows = cur.fetchall()
        for r in rows:
            print(f"[{r[4]}] Key: {r[0]} | Kind: {r[2]} | Project: {r[3]}")
            print(f"Value: {r[1]}")
            print("-" * 50)
            
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error querying memory database: {e}", file=sys.stderr)

if __name__ == "__main__":
    query_memory()
