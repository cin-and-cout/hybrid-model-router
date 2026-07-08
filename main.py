import sys
import httpx
import pydantic

def main():
    print("==================================================")
    print("Hello from the Hybrid Model Router Agent!")
    print(f"Python Version: {sys.version}")
    print("Dependencies verification:")
    print(f"  - httpx: {httpx.__version__}")
    print(f"  - pydantic: {pydantic.__version__}")
    print("Docker environment set up successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
