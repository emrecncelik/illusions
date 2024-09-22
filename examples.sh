# Create stimuli set for warren1968 with 20ms gaps
# with 50, 100, 200, and 300 repetitions. Silent parts 
# of the beggining and end of the stimuli are removed.
# python run_stimuli_generation.py \
#     --repetitions 50 100 200 300 \
#     --gap 20 \
#     --remove_silent_edges

# Same for natsoulas1965 stimuli
python run_stimuli_generation.py \
    --stimuli natsoulas1965 \
    --repetitions 50 100 200 300 \
    --gap 20 \
    --remove_silent_edges