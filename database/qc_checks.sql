-- Null checks
SELECT 'null_gene_symbol' AS check_name, COUNT(*) AS n_issues FROM genes WHERE symbol IS NULL OR symbol = '';
SELECT 'null_disease_name' AS check_name, COUNT(*) AS n_issues FROM diseases WHERE disease_name IS NULL OR disease_name = '';

-- Duplicate checks
SELECT 'duplicate_gene_symbol' AS check_name, COUNT(*) AS n_issues
FROM (
    SELECT symbol FROM genes GROUP BY symbol HAVING COUNT(*) > 1
);

SELECT 'duplicate_disease_name' AS check_name, COUNT(*) AS n_issues
FROM (
    SELECT disease_name FROM diseases GROUP BY disease_name HAVING COUNT(*) > 1
);

-- Orphan checks
SELECT 'orphan_edges_gene' AS check_name, COUNT(*) AS n_issues
FROM atlas_edges e LEFT JOIN genes g ON e.gene_id = g.gene_id
WHERE g.gene_id IS NULL;

SELECT 'orphan_edges_disease' AS check_name, COUNT(*) AS n_issues
FROM atlas_edges e LEFT JOIN diseases d ON e.disease_id = d.disease_id
WHERE d.disease_id IS NULL;

SELECT 'orphan_edges_source' AS check_name, COUNT(*) AS n_issues
FROM atlas_edges e LEFT JOIN provenance_sources s ON e.source_id = s.source_id
WHERE s.source_id IS NULL;
