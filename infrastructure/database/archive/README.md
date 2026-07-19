# Archived schema files

Pre-consolidation SQL artifacts. **Not loaded by any code path.** Kept
in tree for historical reference only; the source of truth for the v1
schema is [`../complete_schema.sql`](../complete_schema.sql) (Part A
base + Part B auto-regenerated from `migrations/`).

| File | Original purpose | Superseded by |
|---|---|---|
| `admin_schema_updates.sql` | Admin role columns + audit log | Part B PR migrations |
| `enhanced_schema_updates.sql` | Mixed schema patches | Part B PR migrations |
| `p2_indexes_and_constraints.sql` | P2 perf indexes | Migration `2026-03-*` series |
| `production_migrations.sql` | Pre-launch hardening | Part B PR migrations |

If you need to apply anything from here to a fresh project, run
`complete_schema.sql` first — these files are likely already covered.
