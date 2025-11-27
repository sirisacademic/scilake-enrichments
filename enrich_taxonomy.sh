#!/bin/bash
# Enrich taxonomy with Wikidata IDs and aliases

set -e

# Default values
DOMAIN=""
USE_GENRE=false
THRESHOLD=0.7
DELAY=0.25
NO_RESUME=false
SIMILARITY_MODEL="intfloat/multilingual-e5-base"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --use-genre)
            USE_GENRE=true
            shift
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --delay)
            DELAY="$2"
            shift 2
            ;;
        --similarity-model)
            SIMILARITY_MODEL="$2"
            shift 2
            ;;
        --no-resume)
            NO_RESUME=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate domain
if [ -z "$DOMAIN" ]; then
    echo "Usage: $0 --domain <energy|maritime|neuro|ccam> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --domain            Domain to enrich (energy, maritime, neuro, ccam)"
    echo "  --use-genre         Enable GENRE model (slower, more accurate)"
    echo "  --threshold N       Similarity threshold 0.0-1.0 (default: 0.6)"
    echo "  --delay N           Delay between API calls in seconds (default: 0.5)"
    echo "  --similarity-model  Embedding model (default: intfloat/multilingual-e5-base)"
    echo "  --no-resume         Start from scratch, ignore checkpoints"
    echo ""
    echo "Examples:"
    echo "  $0 --domain energy"
    echo "  $0 --domain maritime --use-genre --threshold 0.7"
    exit 1
fi

# Set paths based on domain
case $DOMAIN in
    energy)
        INPUT="taxonomies/energy/IRENA.tsv"
        OUTPUT="taxonomies/energy/IRENA_enriched.tsv"
        ;;
    maritime)
        INPUT="taxonomies/maritime/VesselTypes.tsv"
        OUTPUT="taxonomies/maritime/VesselTypes_enriched.tsv"
        ;;
    neuro)
        INPUT="taxonomies/neuro/Neuroscience_Combined.tsv"
        OUTPUT="taxonomies/neuro/Neuroscience_Combined_enriched.tsv"
        ;;
    ccam)
        INPUT="taxonomies/ccam/CCAM_Combined.tsv"
        OUTPUT="taxonomies/ccam/CCAM_Combined_enriched.tsv"
        ;;
    *)
        echo "❌ Unknown domain: $DOMAIN"
        echo "   Valid domains: energy, maritime, neuro, ccam"
        exit 1
        ;;
esac

# Check input exists
if [ ! -f "$INPUT" ]; then
    echo "❌ Input file not found: $INPUT"
    exit 1
fi

# Build command
CMD="python scripts/wikidata_enricher.py --input $INPUT --output $OUTPUT --threshold $THRESHOLD --delay $DELAY --similarity-model $SIMILARITY_MODEL"

if [ "$USE_GENRE" = true ]; then
    CMD="$CMD --use-genre"
fi

if [ "$NO_RESUME" = true ]; then
    CMD="$CMD --no-resume"
fi

# Run enrichment
echo "════════════════════════════════════════════════════════════════"
echo "Wikidata Taxonomy Enrichment"
echo "════════════════════════════════════════════════════════════════"
echo "Domain:           $DOMAIN"
echo "Input:            $INPUT"
echo "Output:           $OUTPUT"
echo "Threshold:        $THRESHOLD"
echo "Similarity Model: $SIMILARITY_MODEL"
echo "GENRE:            $USE_GENRE"
echo "════════════════════════════════════════════════════════════════"
echo ""

eval $CMD

echo ""
echo "✅ Enrichment complete!"
echo "   Output saved to: $OUTPUT"
