
import os

f = r'backend\static\index.html'
print(f"--- Checking {f} ---")
with open(f, 'rb') as fp:
    content = fp.read()
    print(f"Size: {len(content)} bytes")
    try:
        text = content.decode('utf-8')
        print("Successfully decoded as UTF-8.")
        print("First 200 chars:")
        print(text[:200])
        print("\nLast 200 chars:")
        print(text[-200:])
    except UnicodeDecodeError:
        print("ERROR: Not valid UTF-8.")
