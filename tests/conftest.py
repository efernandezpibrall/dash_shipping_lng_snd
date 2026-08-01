import os


# Exercise the new implementation in the normal suite while production
# defaults remain disabled for the requested staged rollout.
os.environ.setdefault("DASH_REVISION_AWARE_REFRESH_ENABLED", "1")
os.environ.setdefault("DASH_FLEET_ARROW_SOURCE_ENABLED", "1")
os.environ.setdefault("DASH_FLEET_RENDER_SNAPSHOT_ENABLED", "1")
os.environ.setdefault("DASH_FLEET_STAGED_RENDER_ENABLED", "1")
