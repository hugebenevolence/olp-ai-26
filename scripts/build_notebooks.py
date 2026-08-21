from olp_ai_26.notebooks import build_all_notebooks

if __name__ == "__main__":
    for output in build_all_notebooks():
        print(f"Wrote {output}")
