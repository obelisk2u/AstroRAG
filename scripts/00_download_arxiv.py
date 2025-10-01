# scripts/00_download_arxiv.py
import pathlib, json, time, sys
import requests, feedparser, yaml
from urllib.parse import urlencode
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs" / "data.yaml"

ARXIV_API = "https://export.arxiv.org/api/query"

def load_cfg():
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("categories", [
        "astro-ph.CO","astro-ph.GA","astro-ph.EP","astro-ph.HE",
        "astro-ph.IM","astro-ph.SR","gr-qc","physics.comp-ph","physics.space-ph"
    ])
    cfg.setdefault("per_category_max", 15000)
    cfg.setdefault("out_jsonl", "data/raw/arxiv_astro.jsonl")
    cfg.setdefault("page_size", 100)           # 50–100 is polite
    cfg.setdefault("delay_seconds", 4)         # delay between requests
    cfg.setdefault("max_empty_skips", 10)      # how many empty pages to skip before giving up on a category
    cfg.setdefault("timeout_seconds", 30)
    cfg.setdefault("debug", False)
    cfg.setdefault("debug_per_category_max", 500)
    return cfg

def open_out_and_seen(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rid = json.loads(line)["id"]
                    if rid:
                        seen.add(rid)
                except Exception:
                    continue
    out = path.open("a", encoding="utf-8")
    return out, seen

def build_url(query: str, start: int, max_results: int):
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return ARXIV_API + "?" + urlencode(params)

def fetch_page(query: str, start: int, page_size: int, timeout: int):
    url = build_url(query, start, page_size)
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "AstroRerank/1.0 (mailto:your.email@example.com)"})
    r.raise_for_status()
    feed = feedparser.parse(r.text)
    return feed

def parse_entries(feed):
    entries = []
    for e in feed.entries:
        # entry_id is a URL like http://arxiv.org/abs/XXXX. We keep it as ID.
        rid = getattr(e, "id", None) or getattr(e, "link", None)
        title = (getattr(e, "title", "") or "").replace("\n", " ").strip()
        summary = (getattr(e, "summary", "") or "").replace("\n", " ").strip()
        published = getattr(e, "published", None)
        # authors
        authors = []
        for a in getattr(e, "authors", []) or []:
            name = a.get("name") if isinstance(a, dict) else getattr(a, "name", None)
            if name: authors.append(name)
        # categories (tags)
        categories = []
        for t in getattr(e, "tags", []) or []:
            term = t.get("term") if isinstance(t, dict) else getattr(t, "term", None)
            if term: categories.append(term)
        if rid:
            entries.append({
                "id": rid,
                "title": title,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "published": published
            })
    return entries

def harvest_category(cat: str, limit: int, page_size: int, delay_s: int, timeout: int, out_f, seen: set, max_empty_skips: int):
    """Fetch up to `limit` records for one category. Skip empty pages and resume by `seen` IDs."""
    written = 0
    start = 0
    empty_skips = 0
    pbar = tqdm(total=limit, desc=f"{cat:15s}", leave=False)

    while written < limit:
        try:
            feed = fetch_page(f"cat:{cat}", start, page_size, timeout)
        except Exception as e:
            # transient network issue; wait and retry same page
            print(f"[warn] {cat}: HTTP error at start={start}: {e}. retrying in {delay_s}s", file=sys.stderr)
            time.sleep(delay_s)
            continue

        entries = parse_entries(feed)

        if not entries:
            empty_skips += 1
            if empty_skips > max_empty_skips:
                print(f"[error] {cat}: too many empty pages (start around {start}); stopping this category.", file=sys.stderr)
                break
            # skip this page offset; advance to next page window
            start += page_size
            time.sleep(delay_s)
            continue

        empty_skips = 0  # reset after a successful non-empty page

        # write new entries
        for rec in entries:
            rid = rec["id"]
            if rid in seen:
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            seen.add(rid)
            written += 1
            pbar.update(1)
            if written >= limit:
                break

        # next page
        start += page_size
        time.sleep(delay_s)

    pbar.close()
    return written

def main():
    cfg = load_cfg()
    per_max = cfg["debug_per_category_max"] if cfg["debug"] else cfg["per_category_max"]
    out_path = ROOT / cfg["out_jsonl"]
    out_f, seen = open_out_and_seen(out_path)

    print(f"Saving to: {out_path}")
    print(f"Categories: {', '.join(cfg['categories'])}")
    print(f"Per-category max: {per_max}")
    print(f"Resume mode: found {len(seen)} existing records")

    total_written = 0
    try:
        for cat in cfg["categories"]:
            wrote = harvest_category(
                cat=cat,
                limit=per_max,
                page_size=cfg["page_size"],
                delay_s=cfg["delay_seconds"],
                timeout=cfg["timeout_seconds"],
                out_f=out_f,
                seen=seen,
                max_empty_skips=cfg["max_empty_skips"],
            )
            total_written += wrote
            print(f"[info] {cat}: wrote {wrote}, total so far {total_written}")
    except KeyboardInterrupt:
        print("\n[info] interrupted; partial file preserved.")
    finally:
        out_f.close()

    print(f"✅ Done. Total unique records in file now: {len(seen)}")

if __name__ == "__main__":
    main()
