import psycopg2
import sys

def insert_memory():
    try:
        conn = psycopg2.connect(
            host="10.0.0.85",
            port="5434",
            database="kruschdb",
            user="openclaw",
            password="openclaw_password"
        )
        cur = conn.cursor()
        
        print("Inserting activity log...")
        cur.execute("SELECT author_id FROM homelab_memory_v2 WHERE author_id IS NOT NULL LIMIT 1")
        author_id = cur.fetchone()[0]
        
        cur.execute("""
            INSERT INTO homelab_memory_v2 (category, content, project, author_id)
            VALUES (%s, %s, %s, %s)
        """, (
            'activity', 
            '[krusch-agentic-proxy] Completed v0.3.0 release. Genericized documentation, removed AMD references, and pushed to origin.', 
            'krusch-agentic-proxy',
            author_id
        ))
        
        print("Inserting steering fact...")
        cur.execute("""
            INSERT INTO ide_agent_nuggets (key, value, kind, project)
            VALUES (%s, %s, %s, %s)
        """, (
            'krusch-agentic-proxy:genericization', 
            'To maintain open-source sanctity, the krusch-agentic-proxy repository must remain free of private homelab network definitions like kruschdev or kruschdb.', 
            'project', 
            'krusch-agentic-proxy'
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        print("Successfully saved memory state.")
    except Exception as e:
        print(f"Error querying memory database: {e}", file=sys.stderr)

if __name__ == "__main__":
    insert_memory()
