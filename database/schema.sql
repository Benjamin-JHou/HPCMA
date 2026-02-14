PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS provenance_sources (
    source_id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL UNIQUE,
    source_version TEXT NOT NULL,
    download_date TEXT NOT NULL,
    checksum_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS genes (
    gene_id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS diseases (
    disease_id INTEGER PRIMARY KEY,
    disease_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS cell_types (
    cell_type_id INTEGER PRIMARY KEY,
    cell_type_name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS atlas_edges (
    edge_id INTEGER PRIMARY KEY,
    gene_id INTEGER NOT NULL,
    disease_id INTEGER NOT NULL,
    cell_type_id INTEGER,
    mr_beta REAL,
    pph4 REAL,
    mechanism_axis TEXT,
    clinical_intervention TEXT,
    priority_score REAL,
    total_influence REAL,
    evidence_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    UNIQUE(gene_id, disease_id, cell_type_id, source_id),
    FOREIGN KEY (gene_id) REFERENCES genes(gene_id),
    FOREIGN KEY (disease_id) REFERENCES diseases(disease_id),
    FOREIGN KEY (cell_type_id) REFERENCES cell_types(cell_type_id),
    FOREIGN KEY (source_id) REFERENCES provenance_sources(source_id)
);

CREATE VIEW IF NOT EXISTS gene_profile_view AS
SELECT
    g.symbol AS gene,
    d.disease_name AS disease,
    COALESCE(c.cell_type_name, '') AS celltype,
    e.evidence_type,
    e.source_id
FROM atlas_edges e
JOIN genes g ON g.gene_id = e.gene_id
JOIN diseases d ON d.disease_id = e.disease_id
LEFT JOIN cell_types c ON c.cell_type_id = e.cell_type_id;
