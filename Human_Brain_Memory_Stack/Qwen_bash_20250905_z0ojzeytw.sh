# Apply this patch to Cogniee v0.8+
sed -i 's/def fuse(self, primary):/def fuse(self, primary, temporal_context=None, salience=0.5):/' cogniee/core.py