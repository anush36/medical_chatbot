import sys
from backend.pmc_tool import get_pmcids_from_query, get_bioc_content

pmcids = get_pmcids_from_query("acute myeloid leukemia", max_results=1)
if pmcids:
    print(pmcids)
    print("---")
    print(get_bioc_content(pmcids[0])[:500])
