-- Disease-centric network query
SELECT d.disease_name AS disease,
       g.symbol AS gene,
       COALESCE(c.cell_type_name, '') AS celltype,
       e.mr_beta,
       e.pph4,
       e.mechanism_axis,
       e.evidence_type,
       e.source_id
FROM atlas_edges e
JOIN diseases d ON d.disease_id = e.disease_id
JOIN genes g ON g.gene_id = e.gene_id
LEFT JOIN cell_types c ON c.cell_type_id = e.cell_type_id
WHERE d.disease_name = :disease_name
ORDER BY e.total_influence DESC;
