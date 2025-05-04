#!/usr/bin/env bash
set -euo pipefail

BUCKET_WIDTH="0.2"
DATA_FILE="./train/training_data.csv"
BUCKETED_DATA="./train/bucketed_entropy.csv"
OUTPUT_PNG="./train/entropy_histogram.png"

if [ ! -f "$DATA_FILE" ]; then
    echo "❌ Data file not found: $DATA_FILE"
    exit 1
fi

echo "▶ Bucketing entropy data with width = $BUCKET_WIDTH…"

# AWK doesn't support floating-point modulus, so we scale integers
SCALE=$(awk -v w="$BUCKET_WIDTH" 'BEGIN { print 1 / w }')

awk -F, -v scale="$SCALE" -v width="$BUCKET_WIDTH" 'NR > 1 {
    bucket_raw = int($2 * scale)
    bucket = bucket_raw / scale
    sum[bucket] += $3
    count[bucket] += 1
}
END {
    for (b in sum) {
        printf("%.2f,%.4f\n", b, sum[b] / count[b])
    }
}' "$DATA_FILE" | sort -n > "$BUCKETED_DATA"

echo "▶ Plotting histogram to $OUTPUT_PNG…"


gnuplot <<-GNUPLOT
    set datafile separator comma
    set terminal pngcairo size 1000,600 enhanced font 'Verdana,10'
    set output "$OUTPUT_PNG"
    set title "Average Moves Remaining by Entropy Bucket"
    set xlabel "Entropy Bucket (width = $BUCKET_WIDTH)"
    set ylabel "Average Moves Remaining"
    set xtics rotate by -45 font ",8"
    set style data histograms
    set style fill solid 1.0 border -1
    set boxwidth 0.2
    set grid ytics

    plot "$BUCKETED_DATA" using 2:xtic(1) title "Avg. Moves"
GNUPLOT

echo "✅ Done. Histogram saved to $OUTPUT_PNG"
