import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.report_content import build_executive_summary_qa
from scripts.report_sections import (
    get_title_markdown, get_executive_summary_markdown,
    get_introduction_markdown, get_dataset_markdown,
    get_features_markdown, get_architecture_markdown,
    get_base_models_markdown, get_autoencoder_markdown,
    get_stacking_markdown, get_fusion_markdown
)
from scripts.report_sprints import get_sprint7_markdown, get_sprint8_markdown, get_sprint9_markdown, get_sprint10_markdown, get_sprint11_markdown, get_sprint12_markdown, get_sprint13_markdown
from scripts.report_syntheses import get_why_it_happened_markdown, get_problems_and_resolutions_markdown, get_consolidated_results_markdown, get_findings_and_conclusion_markdown, get_appendices_markdown, get_final_checklist_markdown

all_texts = [
    get_title_markdown(), get_executive_summary_markdown(build_executive_summary_qa()),
    get_introduction_markdown(), get_dataset_markdown(), get_features_markdown(),
    get_architecture_markdown(), get_base_models_markdown(), get_autoencoder_markdown(),
    get_stacking_markdown(), get_fusion_markdown(),
    get_sprint7_markdown(), get_sprint8_markdown(), get_sprint9_markdown(),
    get_sprint10_markdown(), get_sprint11_markdown(), get_sprint12_markdown(),
    get_sprint13_markdown(),
    get_why_it_happened_markdown(), get_problems_and_resolutions_markdown(),
    get_consolidated_results_markdown(), get_findings_and_conclusion_markdown(),
    get_appendices_markdown(), get_final_checklist_markdown()
]

combined = '\n'.join(all_texts)
math_matches = re.findall(r'\$(.+?)\$', combined)
print(f'Total math blocks: {len(math_matches)}')
unique_math = sorted(set(math_matches))
for m in unique_math:
    print('  MATH:', repr(m))
