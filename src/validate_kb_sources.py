
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import workloadlens_kb as kb  # reuses read_pdf / read_html / clean_text — no embedding model needed


def run():
    meta = pd.read_excel(kb.META_FILE, sheet_name="Documents").fillna("")
    expected = meta[meta["Status"] != "Drop"]
    expected_ids = set(expected["ID"].astype(int))
    dropped_ids = set(meta[meta["Status"] == "Drop"]["ID"].astype(int))

    all_files = list(kb.DOCS_DIR.glob("*.*"))
    # group files by their ID (the number before the extension)
    files_by_id = {}
    stray_files = []
    for f in all_files:
        try:
            file_id = int(f.stem)
        except ValueError:
            stray_files.append((f.name, "filename isn't a plain number — not matchable to any spreadsheet ID"))
            continue
        files_by_id.setdefault(file_id, []).append(f)

    print(f"Spreadsheet: {len(expected_ids)} expected documents (Status != Drop), "
          f"{len(dropped_ids)} marked Drop\n")
    print(f"docs/ folder: {len(all_files)} files found\n")

    # 1. missing: expected but no file
    missing = sorted(expected_ids - files_by_id.keys())
    if missing:
        print(f"MISSING ({len(missing)}) — spreadsheet expects these, no file found:")
        for doc_id in missing:
            title = expected[expected["ID"] == doc_id]["Title"].iloc[0]
            print(f"  ID {doc_id}: {title[:70]}")
        print()

    # 2. stray: file present but ID not in spreadsheet at all, or is a Dropped ID
    stray_ids = sorted(set(files_by_id.keys()) - expected_ids - dropped_ids)
    if stray_ids:
        print(f"STRAY ({len(stray_ids)}) — file(s) present but this ID isn't in the spreadsheet "
              f"as a usable document:")
        for doc_id in stray_ids:
            for f in files_by_id[doc_id]:
                print(f"  {f.name} (ID {doc_id} not found in spreadsheet)")
        print()

    dropped_but_present = sorted(set(files_by_id.keys()) & dropped_ids)
    if dropped_but_present:
        print(f"DROPPED BUT PRESENT ({len(dropped_but_present)}) — these IDs are marked "
              f"Status=Drop in your spreadsheet, but you have a file for them anyway "
              f"(build() will skip them regardless, but worth knowing why you have them):")
        for doc_id in dropped_but_present:
            for f in files_by_id[doc_id]:
                print(f"  {f.name}")
        print()

    # 3. duplicates: more than one file for the same ID
    duplicates = {doc_id: files for doc_id, files in files_by_id.items() if len(files) > 1}
    if duplicates:
        print(f"DUPLICATE IDs ({len(duplicates)}) — more than one file for the same ID. "
              f"build() will silently use only ONE of these and ignore the rest:")
        for doc_id, files in duplicates.items():
            names = ", ".join(f.name for f in files)
            print(f"  ID {doc_id}: {names}  <- pick one, delete/rename the other(s)")
        print()

    if stray_files:
        print(f"UNPARSEABLE FILENAMES ({len(stray_files)}):")
        for name, reason in stray_files:
            print(f"  {name}: {reason}")
        print()

    # 4. text-extraction check for every matched, non-dropped file
    print("Text extraction check (this reads every file — may take a moment):")
    thin_files = []
    for doc_id in sorted(expected_ids & files_by_id.keys()):
        f = files_by_id[doc_id][0]  # same "pick first" behaviour as build()
        try:
            text = kb.clean_text(kb.read_doc(f))
            n_chars = len(text)
            status = "OK" if n_chars >= 500 else "THIN - likely scanned/login page"
            print(f"  ID {doc_id:>2} ({f.name}): {n_chars:>6} characters — {status}")
            if n_chars < 500:
                thin_files.append((doc_id, f.name))
        except Exception as e:
            print(f"  ID {doc_id:>2} ({f.name}): FAILED to read — {e}")
            thin_files.append((doc_id, f.name))

    print(f"\n=== SUMMARY ===")
    print(f"Usable now: {len(expected_ids & files_by_id.keys()) - len(thin_files)} / {len(expected_ids)}")
    if missing:
        print(f"Missing: {len(missing)} (listed above)")
    if thin_files:
        print(f"Thin/unreadable: {len(thin_files)} (listed above) — these will still 'work' "
              f"(build() won't crash) but will produce near-useless chunks in your KB")
    if duplicates:
        print(f"Duplicate IDs needing your attention: {len(duplicates)}")
    if not missing and not thin_files and not duplicates:
        print("Everything checks out — safe to run the full build.")


if __name__ == "__main__":
    run()
