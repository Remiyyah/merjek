#!/bin/bash
#SBATCH --job-name=olam-gen
#SBATCH --output=olam-output.txt
#SBATCH --error=olam-error.txt
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00


export OLLAMA_HOST=http://0.0.0.0:11434      # All connections

apptainer run --nv --bind /project/jmflagg/ollama/data:/data --bind /project/jmflagg/ollama:/ollama  ollama_latest.sif serve &

# Give the container a few seconds to start up
sleep 5

# Confirm Ollama is accessible
echo "📝 Checking Ollama connection..."
curl -s $OLLAMA_HOST || { echo "❗ Failed to connect to Ollama. Ensure it is running on the node. Exiting."; exit 1; }

# Activate virtual environment
source /project/jmflagg/ollama/ollama_env/bin/activate  # Install ollama and pymongo libraries with pip in your venv

# Run the Python script
echo "🚀 Starting document processing with Ollama..."
python /project/jmflagg/ollama/cluster_gen2.py          # Whatever your python file name is

# Deactivate virtual environment after the job is done
deactivate
echo "✅ Document processing job completed."