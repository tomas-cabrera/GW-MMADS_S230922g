#/bin/bash

### Variables
# Environment name
ENV_NAME="GW-MMADS_S230922g"

# Activate environment
conda activate $ENV_NAME

# Export environment
conda env export > "$(dirname "$0")/environment.yml" 
