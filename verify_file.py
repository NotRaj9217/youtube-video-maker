
import os

files = [
    r'backend\static\script_backup.js',
    r'backend\static\script.js',
    r'backend\static\index.html'
]

for f in files:
    print(f"--- Checking {f} ---")
    if not os.path.exists(f):
        print("File not found.")
        continue
    
    try:
        with open(f, 'rb') as fp:
            content = fp.read()
            print(f"Size: {len(content)} bytes")
            # Check for null bytes
            if b'\0' in content:
                print("WARNING: Null bytes found (binary file?)")
            
            # Try decoding
            try:
                text = content.decode('utf-8')
                print("Successfully decoded as UTF-8.")
                print("First 200 chars:")
                print(text[:200])
                print("\nLast 200 chars:")
                print(text[-200:])
            except UnicodeDecodeError:
                print("ERROR: Not valid UTF-8.")
    except Exception as e:
        print(f"Error reading file: {e}")
    print("\n")
