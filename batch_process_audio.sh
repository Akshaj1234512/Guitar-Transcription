#!/bin/bash
# Batch processing script for TART guitar transcription
# Processes all audio files in multiple directories

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <audio_dir1> [audio_dir2] [audio_dir3] ... [--output output_dir]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/audio1 /path/to/audio2 --output ./results"
    exit 1
fi

OUTPUT_DIR="./results"
AUDIO_DIRS=()

# Parse arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            AUDIO_DIRS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "BATCH PROCESSING MULTIPLE DIRECTORIES"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
echo "Directories to process: ${#AUDIO_DIRS[@]}"
echo ""

# Overall log file
OVERALL_LOG="$OUTPUT_DIR/batch_overall_$(date +%Y%m%d_%H%M%S).log"
FAILED_SUMMARY="$OUTPUT_DIR/failed_summary_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "Batch processing started at $(date)"
    echo "Total directories: ${#AUDIO_DIRS[@]}"
    echo "---"
} > "$OVERALL_LOG"

TOTAL_PROCESSED=0
TOTAL_FAILED=0

# Process each directory
for idx in "${!AUDIO_DIRS[@]}"; do
    AUDIO_DIR="${AUDIO_DIRS[$idx]}"
    DIR_IDX=$((idx + 1))
    
    if [ ! -d "$AUDIO_DIR" ]; then
        echo "[$DIR_IDX/${#AUDIO_DIRS[@]}] Error: Directory does not exist: $AUDIO_DIR" | tee -a "$OVERALL_LOG"
        continue
    fi
    
    DIR_NAME=$(basename "$AUDIO_DIR")
    DIR_OUTPUT="$OUTPUT_DIR/$DIR_NAME"
    mkdir -p "$DIR_OUTPUT"
    
    echo "[$DIR_IDX/${#AUDIO_DIRS[@]}] Processing: $AUDIO_DIR"
    
    # Count audio files
    TOTAL=$(find "$AUDIO_DIR" -maxdepth 1 \( -name "*.wav" -o -name "*.mp3" -o -name "*.flac" \) | wc -l)
    
    if [ "$TOTAL" -eq 0 ]; then
        echo "  Warning: No audio files found in $AUDIO_DIR" | tee -a "$OVERALL_LOG"
        continue
    fi
    
    echo "  Found $TOTAL audio files"
    
    # Create log file for this directory
    LOG_FILE="$DIR_OUTPUT/batch_process_$(date +%Y%m%d_%H%M%S).log"
    
    {
        echo "Directory processing started at $(date)"
        echo "Source: $AUDIO_DIR"
        echo "Total files: $TOTAL"
        echo "---"
    } > "$LOG_FILE"
    
    PROCESSED=0
    FAILED=0
    
    # Process each audio file
    for audio_file in "$AUDIO_DIR"/*.{wav,mp3,flac}; do
        # Skip if file doesn't exist (glob didn't match)
        [ -e "$audio_file" ] || continue
        
        PROCESSED=$((PROCESSED + 1))
        filename=$(basename "$audio_file")
        
        printf "    [%d/%d] %s ... " "$PROCESSED" "$TOTAL" "$filename"
        
        if python predict.py --audio_path "$audio_file" >> "$LOG_FILE" 2>&1; then
            echo "✓"
        else
            echo "✗"
            echo "$DIR_NAME: $filename" >> "$FAILED_SUMMARY"
            FAILED=$((FAILED + 1))
        fi
    done
    
    echo "  Results: $((PROCESSED - FAILED)) successful, $FAILED failed"
    echo "  Log: $LOG_FILE" | tee -a "$OVERALL_LOG"
    
    TOTAL_PROCESSED=$((TOTAL_PROCESSED + PROCESSED))
    TOTAL_FAILED=$((TOTAL_FAILED + FAILED))
    echo "" | tee -a "$OVERALL_LOG"
done

echo ""
echo "============================================================"
echo "OVERALL BATCH PROCESSING COMPLETE"
echo "============================================================"
echo "Total files processed: $TOTAL_PROCESSED"
echo "Successful: $((TOTAL_PROCESSED - TOTAL_FAILED))"
echo "Failed: $TOTAL_FAILED"
echo "Output directory: $OUTPUT_DIR"
[ "$TOTAL_FAILED" -gt 0 ] && echo "Failed summary: $FAILED_SUMMARY"
echo "Overall log: $OVERALL_LOG"
echo ""

