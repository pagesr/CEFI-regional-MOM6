"""Central paths for the standalone forecast_multi_proc workflow."""

import os
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[3]  # tools/forecast_multi_proc
STANDALONE_ROOT = WORKFLOW_ROOT / "standalone"
FORECAST_CGOA_DIR = STANDALONE_ROOT / "forecast_cgoa"

INITIAL_DIR = STANDALONE_ROOT / "initial"
PHY_OBC_DIR = STANDALONE_ROOT / "boundary" / "PHY"
BGC_OBC_DIR = STANDALONE_ROOT / "boundary" / "BGC"

IC_PHY_SCRIPT = INITIAL_DIR / "nep_to_goa_phy_ic.py"
IC_BGC_SCRIPT = INITIAL_DIR / "nep_to_goa_bgc_ic.py"
PHY_OBC_SCRIPT = PHY_OBC_DIR / "write_CGOA_boundary_2Dfrc-padded.py"
BGC_OBC_SCRIPT = BGC_OBC_DIR / "OBC_BGC.py"
BGC_OBC_POSTPROCESS_SCRIPT = FORECAST_CGOA_DIR / "postprocess_bgc_obc_nco.sh"

DEFAULT_OUTPUT_ROOT = WORKFLOW_ROOT / "outputs"
DEFAULT_CONFIG_ROOT = WORKFLOW_ROOT / "generated_configs"
DEFAULT_LOG_ROOT = Path(os.environ.get("FORECAST_LOG_ROOT", str(WORKFLOW_ROOT / "logs")))
DEFAULT_TEMPLATE_ROOT = FORECAST_CGOA_DIR / "config_templates"
