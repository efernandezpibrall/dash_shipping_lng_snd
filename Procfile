web: exec gunicorn index_shipping_snd:server --bind ${DASH_BIND_HOST:-127.0.0.1}:${PORT:-8067} --workers ${DASH_WEB_WORKERS:-4} --threads ${DASH_WEB_THREADS:-1} --timeout ${DASH_WEB_TIMEOUT:-180}
