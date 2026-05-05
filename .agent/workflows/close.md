---
description: Pause krusch-agentic-mcp and save semantic state
---

# /close — krusch-agentic-mcp

## Steps

1. **Semantic Snapshot**:
   ```bash
   node /home/kruschdev/homelab/projects/pg-git/scripts/sync_to_pg.js /home/kruschdev/homelab/projects/krusch-agentic-mcp
   ```

2. **Update GEMINI_INFLIGHT.md**:
   - Create or overwrite `GEMINI_INFLIGHT.md` in this project root.
   - Include Fragile files, concrete Next Steps, and any transient state.

3. **Log Activity**:
   - Execute `mcp_homelab-memory_mcp_homelab-memory_add` with `category: 'activity'` and content: `[krusch-agentic-mcp] <description>`.

4. **Save Steering Facts**:
   - Store any new patterns via `mcp_nuggets-memory_remember` with `kind: 'project'`, key prefixed `krusch-agentic-mcp:`.

5. **Summarize**: 
   > "Project state saved. See you next session."
