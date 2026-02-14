# Query Examples and Expected Outputs

## Query 1: Gene Profile

- SQL: `database/queries/query_gene_profile.sql`
- Bind: `gene_symbol = 'ACE'`
- Expected columns (stable contract):
  - `gene`
  - `disease`
  - `celltype`
  - `evidence_type`
  - `source_id`

## Query 2: Disease Network

- SQL: `database/queries/query_disease_network.sql`
- Bind: `disease_name = 'CAD'`
- Expected columns:
  - `disease`, `gene`, `celltype`, `mr_beta`, `pph4`, `mechanism_axis`, `evidence_type`, `source_id`

## Query 3: Cell Type Links

- SQL: `database/queries/query_celltype_links.sql`
- Bind: `cell_type_name = 'Endothelial'`
- Expected columns:
  - `celltype`, `gene`, `disease`, `mechanism_axis`, `evidence_type`, `source_id`
