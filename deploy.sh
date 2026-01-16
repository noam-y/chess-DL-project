#!/bin/bash

echo "🚀 Starting Deployment..."

# 1. העלאת השינויים לגיט
git add .
git commit -m "Auto-update from VSCode"
git push

# 2. התחברות לקלאסטר, משיכת הקוד והרצת העבודה
# שימי לב: הפקודה הזו מתחברת, נכנסת לתיקייה, מושכת קוד ושולחת ל-SLURM
ssh dt-2080-12.auth.ad.bgu.ac.il "cd ~/chess-DL-project && git pull && sbatch submit_job.sh"

echo "✅ Job submitted successfully!"