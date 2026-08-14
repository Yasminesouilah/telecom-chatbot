import requests
import time

url = "http://127.0.0.1:8000/chat/stream"

payload = {"message": "Hello, can you tell me about my mobile plan?"}

with requests.post(url, json=payload, stream=True) as r:
    print('status', r.status_code)
    r.raise_for_status()
    for chunk in r.iter_content(chunk_size=64):
        if chunk:
            print('chunk:', chunk.decode('utf-8'), flush=True)
            time.sleep(0.05)
print('done')
