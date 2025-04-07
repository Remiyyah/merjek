#!/bin/bash
#SBATCH --job-name=app-gen
#SBATCH --output=app-output.txt
#SBATCH --error=app-error.txt
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

echo "Starting SLURM job: $SLURM_JOB_NAME"

# Ensure Ollama is accessible by setting OLLAMA_HOST
 #export OLLAMA_HOST="http://$(hostname -i):11434"
# export OLLAMA_HOST="http://127.0.0.1:11434"
export OLLAMA_HOST="http://0.0.0.0:11434"

echo "Using OLLAMA_HOST=$OLLAMA_HOST"

# Load necessary modules if applicable
# module load cuda/12.2

# Set up Apptainer to use GPU and bind the data directory
echo "🔎 Running Ollama using Apptainer with GPU..."

# Make sure your Apptainer container image (ollama_latest.sif) is in place

 apptainer run --nv --bind /project/jmflagg/ollama/data:/data --bind /project/jmflagg/ollama:/ollama  ollama_latest.sif serve &



sleep 5

# Confirm Ollama is accessible
echo "📝 Checking Ollama connection..."
curl -s $OLLAMA_HOST || { echo "❗ Failed to connect to Ollama. Ensure it is running on the node. Exiting."; exit 1; }

# Activate Python virtual environment
echo "🔎 Activating Python virtual environment..."
source /project/jmflagg/ollama/ollama_env/bin/activate

# Confirm environment
echo "🐍 Python version: $(python --version)"
echo "💡 Pip packages:"
pip list | grep ollama

# Run the Ollama model inside the container
echo "📝 Generating essay with Ollama..."
ollama run llama3.2:3b "Write an essay explaining how to learn how to play 5 different instruments." > essay-result.txt
if [ $? -ne 0 ]; then
  echo "❗ Ollama failed to generate the essay. Check essay-error.txt for details."
  exit 1
fi

echo "✅ Essay generated successfully. Check essay-result.txt."

# Deactivate virtual environment
echo "🛑 Deactivating virtual environment..."
deactivate

# Optionally, stop the container if needed
echo "🛑 Stopping Ollama container..."
kill %1