-- ============================================================================
-- SCHEMA MIGRATIONS TRACKING TABLE
-- Run this FIRST before any other migrations.
-- Tracks which migrations have been applied to prevent duplicates.
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    version TEXT UNIQUE NOT NULL,
    description TEXT,
    applied_at TIMESTAMPTZ DEFAULT NOW()
);

-- Grant access
GRANT ALL ON schema_migrations TO authenticated;
GRANT ALL ON SEQUENCE schema_migrations_id_seq TO authenticated;

-- Record base schema files as already applied
INSERT INTO schema_migrations (version, description) VALUES
    ('001_complete_schema', 'Base production schema — 28 tables'),
    ('002_production_migrations', 'RLS policies, triggers, views'),
    ('003_admin_schema_updates', 'Admin features and permissions'),
    ('004_enhanced_schema_updates', 'AI ensemble fields, regime tracking, model performance'),
    ('005_p2_indexes_and_constraints', 'Composite indexes and UNIQUE constraints'),
    ('006_2026_02_05_prd_alignment', 'Paper/live flags, broker_connections'),
    ('007_2026_02_05_eod_scanner', 'daily_universe + eod_scan_runs tables')
ON CONFLICT (version) DO NOTHING;
