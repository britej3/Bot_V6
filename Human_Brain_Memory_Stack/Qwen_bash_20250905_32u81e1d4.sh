# Install critical patches
pip install letta==0.8.3 graphiti==1.2.1 lightrag==0.4.0
python -m cogniee.patch_temporal_fusion

# Start the stack
python memory_stack.py --warmup 24h --enable-temporal-routing