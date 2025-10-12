#!/bin/bash

echo "=========================================="
echo "Training Progress Monitor"
echo "=========================================="
echo ""

cd /Users/user/API_for_Medical_Imaging/backend/models

while true; do
    clear
    echo "=========================================="
    echo "🚀 TRAINING STATUS - $(date '+%H:%M:%S')"
    echo "=========================================="
    echo ""
    
    # Check if process is running
    if ps aux | grep -E "train_extended_epochs" | grep -v grep > /dev/null; then
        echo "✅ Status: TRAINING IN PROGRESS"
        echo ""
        
        # Show latest epoch info
        echo "📊 Latest Progress:"
        echo "---"
        tail -100 training_extended.log | grep -E "(Epoch [0-9]|Train Loss|Val Loss|Val Acc|completed|best|Early stopping)" | tail -10
        
        echo ""
        echo "---"
        
        # Count completed epochs
        COMPLETED=$(grep -c "Val Acc:" training_extended.log 2>/dev/null || echo "0")
        echo "📈 Epochs completed: $COMPLETED/30 (Experiment 1/6)"
        
        # Check experiment progress
        if [ -d "training_results_extended" ]; then
            RESULTS=$(ls -1 training_results_extended/results_*.json 2>/dev/null | wc -l)
            echo "✓ Experiments completed: $RESULTS/6"
        fi
        
    else
        echo "⏹️  Status: TRAINING NOT RUNNING"
        echo ""
        echo "Last log entries:"
        tail -20 training_extended.log
    fi
    
    echo ""
    echo "=========================================="
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 30 seconds..."
    echo "=========================================="
    
    sleep 30
done




