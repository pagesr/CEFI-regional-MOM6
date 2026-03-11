"""Central paths for the CGOA forecast IC/OBC automation workflow."""

from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = WORKFLOW_ROOT.parents[1]

INITIAL_DIR = REPO_ROOT / "tools" / "initial"
PHY_OBC_DIR = REPO_ROOT / "tools" / "boundary" / "forecast" / "PHY"
BGC_OBC_DIR = REPO_ROOT / "tools" / "boundary" / "forecast" / "BGC"

IC_PHY_SCRIPT = INITIAL_DIR / "nep_to_goa_phy_ic.py"
IC_BGC_SCRIPT = INITIAL_DIR / "nep_to_goa_bgc_ic.py"
PHY_OBC_SCRIPT = PHY_OBC_DIR / "write_CGOA_boundary_2Dfrc-padded.py"
BGC_OBC_SCRIPT = BGC_OBC_DIR / "OBC_BGC.py"

DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs"
DEFAULT_CONFIG_ROOT = WORKFLOW_ROOT / "generated_configs"
DEFAULT_LOG_ROOT = WORKFLOW_ROOT / "logs"
DEFAULT_TEMPLATE_ROOT = WORKFLOW_ROOT / "config_templates"
