import sys
import os

# 1. Force Python to find the main 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

# 2. Add it to the top of Python's search list
sys.path.insert(0, src_dir)

try:
    # Now it will definitely find 'modules'
    import modules.kenneth.kenneth_module
    print("✅ SUCCESS: The code has no errors! You can run the main app now.")
except Exception as e:
    print("❌ HIDDEN ERROR FOUND:")
    import traceback
    traceback.print_exc()

# Step 1: Add a new line to the original code when sending one.