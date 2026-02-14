-- Gene-centric profile query
SELECT gene, disease, celltype, evidence_type, source_id
FROM gene_profile_view
WHERE gene = :gene_symbol
ORDER BY disease, celltype;
