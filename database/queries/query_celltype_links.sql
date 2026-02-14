-- Cell type linkage query
SELECT c.cell_type_name AS celltype,
       g.symbol AS gene,
       d.disease_name AS disease,
       e.mechanism_axis,
       e.evidence_type,
       e.source_id
FROM atlas_edges e
JOIN cell_types c ON c.cell_type_id = e.cell_type_id
JOIN genes g ON g.gene_id = e.gene_id
JOIN diseases d ON d.disease_id = e.disease_id
WHERE c.cell_type_name = :cell_type_name
ORDER BY d.disease_name, g.symbol;
